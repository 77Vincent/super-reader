(function exposeModelBackend(root, factory) {
  const modelData = typeof module === "object" && module.exports
    ? require("../boundary-model-data.js")
    : root.SuperReaderBoundaryModelData;
  const api = factory(modelData);

  if (typeof module === "object" && module.exports) {
    module.exports = api;
  }

  root.SuperReaderModelBackend = api;
})(globalThis, function createModelBackend(modelData) {
  "use strict";

  const tensorDescriptors = modelData && modelData.tensors ? modelData.tensors : {};
  const CHANNELS = modelData && modelData.architecture
    ? modelData.architecture.channels
    : tensorDescriptors["embedding.weight"].shape[1];
  const RESIDUAL_BLOCKS = modelData && modelData.architecture
    ? modelData.architecture.residualBlocks
    : Object.keys(tensorDescriptors)
      .filter((name) => /^blocks\.\d+\.normalization\.weight$/u.test(name))
      .length;
  const FEATURE_CHANNELS = CHANNELS * 4;
  const LAYER_NORM_EPSILON = 1e-5;
  const GELU_FACTOR = Math.sqrt(2 / Math.PI);
  let decodedWeights = null;
  let model = null;

  function decodeBase64(value) {
    if (typeof atob === "function") {
      const binary = atob(value);
      const bytes = new Uint8Array(binary.length);
      for (let index = 0; index < binary.length; index += 1) {
        bytes[index] = binary.charCodeAt(index);
      }
      return bytes;
    }

    return Uint8Array.from(Buffer.from(value, "base64"));
  }

  function getWeights() {
    if (decodedWeights) return decodedWeights;
    if (!modelData || modelData.format !== "super-reader-f32-v1") {
      throw new Error("Super Reader boundary model data is unavailable or incompatible");
    }

    const bytes = decodeBase64(modelData.weightsBase64);
    if (bytes.byteLength % Float32Array.BYTES_PER_ELEMENT !== 0) {
      throw new Error("Super Reader boundary model has an invalid byte length");
    }
    decodedWeights = new Float32Array(
      bytes.buffer,
      bytes.byteOffset,
      bytes.byteLength / Float32Array.BYTES_PER_ELEMENT,
    );
    return decodedWeights;
  }

  function tensor(name) {
    const descriptor = modelData.tensors[name];
    if (!descriptor) throw new Error(`Missing boundary-model tensor: ${name}`);
    return getWeights().subarray(
      descriptor.offset,
      descriptor.offset + descriptor.length,
    );
  }

  function gelu(value) {
    return 0.5 * value * (
      1 + Math.tanh(GELU_FACTOR * (value + 0.044715 * value * value * value))
    );
  }

  function normalize(input, output, sequenceLength, weight, bias) {
    for (let position = 0; position < sequenceLength; position += 1) {
      const offset = position * CHANNELS;
      let mean = 0;
      for (let channel = 0; channel < CHANNELS; channel += 1) {
        mean += input[offset + channel];
      }
      mean /= CHANNELS;

      let variance = 0;
      for (let channel = 0; channel < CHANNELS; channel += 1) {
        const difference = input[offset + channel] - mean;
        variance += difference * difference;
      }
      const inverseDeviation = 1 / Math.sqrt(variance / CHANNELS + LAYER_NORM_EPSILON);

      for (let channel = 0; channel < CHANNELS; channel += 1) {
        output[offset + channel] = (
          (input[offset + channel] - mean) * inverseDeviation * weight[channel] + bias[channel]
        );
      }
    }

    return output;
  }

  function convolve(input, output, sequenceLength, weight, bias, activate) {
    for (let position = 0; position < sequenceLength; position += 1) {
      for (let outputChannel = 0; outputChannel < CHANNELS; outputChannel += 1) {
        let value = bias[outputChannel];
        const outputWeightOffset = outputChannel * CHANNELS * 3;

        for (let inputChannel = 0; inputChannel < CHANNELS; inputChannel += 1) {
          const weightOffset = outputWeightOffset + inputChannel * 3;
          if (position > 0) {
            value += input[(position - 1) * CHANNELS + inputChannel] * weight[weightOffset];
          }
          value += input[position * CHANNELS + inputChannel] * weight[weightOffset + 1];
          if (position + 1 < sequenceLength) {
            value += input[(position + 1) * CHANNELS + inputChannel] * weight[weightOffset + 2];
          }
        }

        output[position * CHANNELS + outputChannel] = activate ? gelu(value) : value;
      }
    }

    return output;
  }

  function createModel() {
    const vocabulary = modelData.vocabulary;
    const embedding = tensor("embedding.weight");
    const blocks = Array.from({ length: RESIDUAL_BLOCKS }, (_, index) => ({
      normalizationWeight: tensor(`blocks.${index}.normalization.weight`),
      normalizationBias: tensor(`blocks.${index}.normalization.bias`),
      firstWeight: tensor(`blocks.${index}.first.weight`),
      firstBias: tensor(`blocks.${index}.first.bias`),
      secondWeight: tensor(`blocks.${index}.second.weight`),
      secondBias: tensor(`blocks.${index}.second.bias`),
      residualScale: tensor(`blocks.${index}.residual_scale`)[0],
    }));
    const boundaryHiddenWeight = tensor("boundary_hidden.weight");
    const boundaryHiddenBias = tensor("boundary_hidden.bias");
    const boundaryOutputWeight = tensor("boundary_output.weight");
    const boundaryOutputBias = tensor("boundary_output.bias")[0];
    const unknownToken = vocabulary["<unk>"];

    function scoreTokens(tokens) {
      if (!Array.isArray(tokens) || tokens.length < 2) return [];

      const sequenceLength = tokens.length;
      let hidden = new Float32Array(sequenceLength * CHANNELS);
      const normalized = new Float32Array(hidden.length);
      const first = new Float32Array(hidden.length);
      const second = new Float32Array(hidden.length);
      for (let position = 0; position < sequenceLength; position += 1) {
        const tokenId = Object.prototype.hasOwnProperty.call(vocabulary, tokens[position])
          ? vocabulary[tokens[position]]
          : unknownToken;
        const embeddingOffset = tokenId * CHANNELS;
        hidden.set(embedding.subarray(embeddingOffset, embeddingOffset + CHANNELS), position * CHANNELS);
      }

      for (const block of blocks) {
        normalize(
          hidden,
          normalized,
          sequenceLength,
          block.normalizationWeight,
          block.normalizationBias,
        );
        convolve(
          normalized,
          first,
          sequenceLength,
          block.firstWeight,
          block.firstBias,
          true,
        );
        convolve(
          first,
          second,
          sequenceLength,
          block.secondWeight,
          block.secondBias,
          false,
        );
        for (let index = 0; index < hidden.length; index += 1) {
          hidden[index] += block.residualScale * second[index];
        }
      }

      const scores = new Float32Array(sequenceLength - 1);
      const features = new Float32Array(FEATURE_CHANNELS);
      const boundaryHidden = new Float32Array(CHANNELS);

      for (let gap = 0; gap < sequenceLength - 1; gap += 1) {
        const leftOffset = gap * CHANNELS;
        const rightOffset = leftOffset + CHANNELS;
        for (let channel = 0; channel < CHANNELS; channel += 1) {
          const left = hidden[leftOffset + channel];
          const right = hidden[rightOffset + channel];
          features[channel] = left;
          features[CHANNELS + channel] = right;
          features[CHANNELS * 2 + channel] = right - left;
          features[CHANNELS * 3 + channel] = right * left;
        }

        for (let outputChannel = 0; outputChannel < CHANNELS; outputChannel += 1) {
          let value = boundaryHiddenBias[outputChannel];
          const weightOffset = outputChannel * FEATURE_CHANNELS;
          for (let inputChannel = 0; inputChannel < FEATURE_CHANNELS; inputChannel += 1) {
            value += features[inputChannel] * boundaryHiddenWeight[weightOffset + inputChannel];
          }
          boundaryHidden[outputChannel] = gelu(value);
        }

        let score = boundaryOutputBias;
        for (let channel = 0; channel < CHANNELS; channel += 1) {
          score += boundaryHidden[channel] * boundaryOutputWeight[channel];
        }
        scores[gap] = score;
      }

      return Array.from(scores);
    }

    return Object.freeze({ scoreTokens });
  }

  function getModel() {
    if (!model) model = createModel();
    return model;
  }

  return Object.freeze({
    getModelInfo() {
      return Object.freeze({
        bestEpoch: modelData.bestEpoch,
        checkpointSha256: modelData.checkpointSha256,
        testAccuracy: modelData.testAccuracy,
        tokenization: modelData.tokenization,
        candidatePositions: modelData.candidatePositions,
        vocabularySize: Object.keys(modelData.vocabulary).length,
        channels: CHANNELS,
        residualBlocks: RESIDUAL_BLOCKS,
        convolutionLayers: RESIDUAL_BLOCKS * 2,
      });
    },
    scoreTokens(tokens) {
      return getModel().scoreTokens(tokens);
    },
  });
});
