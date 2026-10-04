import argparse
import json
import re
import os
import unicodedata
from typing import Tuple, List
from multiprocessing import Pool

import fasttext
import pandas as pd
from tqdm import tqdm
from transformers import LlamaTokenizerFast


language_model_map = {
    "en": "classifiers/ultra_fineweb_en.bin",
    "zh": "classifiers/ultra_fineweb_zh.bin"
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, required=True, help="Inference language, support: en, zh.")
    parser.add_argument("--data-path", type=str, required=True, help="Data path.")
    parser.add_argument("--save-path", type=str, required=True, help="Save path root.")
    parser.add_argument("--content-key", type=str, required=True, help="Content key for inference.")
    parser.add_argument("--tokenizer-path", type=str, default="local_tokenizer", help="Tokenizer path.")
    parser.add_argument("--processes-num", type=int, default=64, help="Number of processes.")
    parser.add_argument("--write-batch-size", type=int, default=100, help="Write batch size.")
    parser.add_argument("--inplace", action="store_true", help="Inplace already processed data.")
    return parser.parse_args()


def fasttext_preprocess_func(content: str, tokenizer: LlamaTokenizerFast) -> str:
    """Fasttext preprocess function.

    Args:
        content (str): Content to process.

    Returns:
        str: Processed normalized content.
    """

    # 1. remove multiple newlines
    content = re.sub(r'\n{3,}', '\n\n', content)

    # 2. lower the content
    content = content.lower()

    # 3. remove diacritics
    content = ''.join(
        c for c in unicodedata.normalize('NFKD', content)
        if unicodedata.category(c) != 'Mn')

    # 4. word segmentation
    token_ids = tokenizer.encode(content, add_special_tokens=False)
    single_text_list = []
    for token_id in token_ids:
        curr_text = tokenizer.decode([token_id])
        single_text_list.append(curr_text)

    content = ' '.join(single_text_list)

    # 5. keep escape chars, \n, \t, \r -> \\n, \\t, \\r,
    # which will saved as \n, \t, \r in txt file.
    content = re.sub(r'\n', '\\\\n', content)
    content = re.sub(r'\r', '\\\\r', content)
    content = re.sub(r'\t', '\\\\t', content)
    content = re.sub(r' +', ' ', content)
    content = content.strip()

    return content


def fasttext_infer(norm_content: str, fasttext_model: fasttext.FastText) -> Tuple[str, float]:
    """Fasttext inference function

    Args:
        content (str): input text
    
    Returns:
        str: json string with pred_label and pred_score
    """

    pred_label, pred_prob = fasttext_model.predict(norm_content)
    pred_label = pred_label[0]
    _score = min(pred_prob.tolist()[0], 1)
    if pred_label == "__label__neg":
        _score = 1 - _score

    return pred_label, _score


def load_data(file_path: str, content_key: str) -> List[str]:
    """Load data from file path.
    
    Args:
        file_path (str): File path.
        content_key (str): Content key.
    
    Returns:
        List[str]: List of content.
    """
    samples = []
    if file_path.endswith(".jsonl") or file_path.endswith(".json"):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                if content_key in data:
                    if data[content_key] == "":
                        print("Empty text, continue")
                        continue
                    if data[content_key] is None:
                        print("None text, continue")
                        continue
                    samples.append(data[content_key])
    elif file_path.endswith(".parquet"):
        df = pd.read_parquet(file_path)
        for _, row in df.iterrows():
            if content_key in row:
                if row[content_key] == "":
                    print("Empty text, continue")
                    continue
                if row[content_key] is None:
                    print("None text, continue")
                    continue
                samples.append(row[content_key])
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
    return samples


def process_file(
        file_path: str,
        tokenizer_path: str,
        fasttext_model_path: str,
        save_path: str,
        item: int,
        content_key: str,
        inplace: bool,
        write_batch_size: int) -> None:
    """Process a single file.
    
    Args:
        file_path (str): File path to process.
        tokenizer_path (str): Tokenizer path.
        fasttext_model_path (str): Fasttext model path.
        save_path (str): Save path.
        item (int): Current process item index.
        content_key (str): Content key.
        write_batch_size (int): Write batch size.
    """

    # load tokenizer and fasttext model
    tokenizer = LlamaTokenizerFast.from_pretrained(tokenizer_path)
    fasttext_model = fasttext.load_model(fasttext_model_path)

    # load data
    all_texts = load_data(file_path, content_key)

    # get file name
    file_name = os.path.basename(file_path)
    curr_file_name = ".".join(file_name.split(".")[:-1])

    output_file = f"{curr_file_name}_fasttext_pos.jsonl"
    output_file = os.path.join(save_path, output_file)

    if inplace and os.path.exists(output_file):
        print(f"File {output_file} already exists, skip")
        return
    
    if os.path.exists(output_file):
        # remove the file
        print(f"File {output_file} already exists, remove it")
        os.remove(output_file)

    results = []
    print(f"ID: {item}, Begin to process {file_path}, total {len(all_texts)} samples, results will be saved in {output_file}")
    for text in tqdm(all_texts):
        norm_content = fasttext_preprocess_func(text, tokenizer)
        label, score = fasttext_infer(norm_content, fasttext_model)

        # label is __label__pos or __label__neg
        if label == "__label__pos":
            curr_result = {"content": text, "pred_label": label, "pred_score": score}
            results.append(curr_result)

        if len(results) >= write_batch_size:
            with open(output_file, "a", encoding="utf-8") as f:
                f.write("\n".join(json.dumps(r, ensure_ascii=False) for r in results) + "\n")
            results.clear()

    # process remaining results
    if results:
        with open(output_file, "a", encoding="utf-8") as f:
            f.write("\n".join(json.dumps(r, ensure_ascii=False) for r in results) + "\n")


def main():
    args = parse_args()
    language = args.language
    data_path = args.data_path
    save_path = args.save_path
    content_key = args.content_key
    tokenizer_path = args.tokenizer_path
    processes_num = args.processes_num
    write_batch_size = args.write_batch_size
    inplace = args.inplace

    assert os.path.exists(data_path), f"Data path {data_path} not exists"
    assert os.path.exists(tokenizer_path), f"Tokenizer path {tokenizer_path} not exists"

    assert language in language_model_map, f"Language {language} not supported"
    fasttext_model_path = language_model_map[language]

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    
    data_path_list = os.listdir(data_path)
    data_path_list = [os.path.join(data_path, file_name) for file_name in data_path_list]

    print("=" * 100)
    print(f"Begin processing\n"
          f"- data path: {data_path}\n"
          f"- save path: {save_path}\n"
          f"- content key: {content_key}\n"
          f"- tokenizer path: {tokenizer_path}\n"
          f"- processes num: {processes_num}\n"
          f"- write batch size: {write_batch_size}\n"
          f"- inplace: {inplace}")
    print("=" * 100)

    print(f"Total {len(data_path_list)} files to process")

    # process data
    with Pool(processes=processes_num) as pool:
        pool.starmap(process_file, [(
            file_path, tokenizer_path, fasttext_model_path, save_path, item, content_key, inplace, write_batch_size)
            for item, file_path in enumerate(data_path_list)])

    print("Finished processing all files")

if __name__ == "__main__":
    main()