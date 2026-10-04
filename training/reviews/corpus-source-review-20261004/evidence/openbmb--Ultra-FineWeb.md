---
language:
- en
- zh
license: apache-2.0
size_categories:
- n>1T
task_categories:
- text-generation
pretty_name: Ultra-FineWeb
tags:
- llm
- pretraining
- web-corpus
- data-filtering
- high-quality
configs:
- config_name: default
  data_files:
  - split: en
    path: data/ultrafineweb_en/*
  - split: zh
    path: data/ultrafineweb_zh/*
  features:
  - name: content
    dtype: string
  - name: score
    dtype: float
  - name: source
    dtype: string
---

# Ultra-FineWeb

<div align="center">
  <img src="assets/ultra-fineweb-logo.png" width="600"/>
</div>

<p align="center">
<a href="https://arxiv.org/abs/2505.05427">📜 Technical Report</a> |
<a href="https://huggingface.co/collections/openbmb/ultradata">📦 UltraData Collection</a> |
<a href="https://ultradata.openbmb.cn/">🌐 UltraData</a> | 
<a href="https://huggingface.co/collections/openbmb/minicpm4">🤗 MiniCPM4 Series</a> |
<a href="https://huggingface.co/collections/openbmb/minicpm5">🤗 MiniCPM5 Series</a>
</p>

<p align="center">
English |
<a href="https://huggingface.co/datasets/openbmb/Ultra-FineWeb/blob/main/README_ZH.md">中文</a>
</p>

## 📚 Introduction

Ultra-FineWeb is a **large-scale, high-quality, and efficiently-filtered dataset**. We use the proposed efficient verification-based high-quality filtering pipeline to the FineWeb and Chinese FineWeb datasets (source data from Chinese FineWeb-edu-v2, which includes IndustryCorpus2, MiChao, WuDao, SkyPile, WanJuan, ChineseWebText, TeleChat, and CCI3), resulting in the creation of higher-quality Ultra-FineWeb-en with approximately 1T tokens, and Ultra-FineWeb-zh datasets with approximately 120B tokens, collectively referred to as Ultra-FineWeb. ***Ultra-FineWeb*** serves as a core pre-training web dataset for the [MiniCPM4 Series](https://huggingface.co/collections/openbmb/minicpm-4-6841ab29d180257e940baa9b) and [MiniCPM5 Series](https://huggingface.co/collections/openbmb/minicpm5) models.

- [Ultra-FineWeb-L1](https://huggingface.co/datasets/openbmb/Ultra-FineWeb-L1): **L1 filtered data** after basic cleaning, heuristic filtering, sensitive-field replacement, and deduplication.
- [Ultra-FineWeb](https://huggingface.co/datasets/openbmb/Ultra-FineWeb): Ultra-FineWeb, a **large-scale, high-quality, and efficiently-filtered dataset**, with 1T English tokens and 120B Chinese tokens. (**Current dataset**)
- [Ultra-FineWeb-classifier](https://huggingface.co/openbmb/Ultra-FineWeb-classifier): Ultra-FineWeb classifier, for filtering high-quality data from web corpora.
- [Ultra-FineWeb-L3](https://huggingface.co/datasets/openbmb/Ultra-FineWeb-L3): the **L3 refined data** based on **Ultra-FineWeb**, via **Q&A Pair Generation** and **Multi-style Rewriting**, with **400B+ English** and **200B+ Chinese** tokens—to our best knowledge, the largest open-source Chinese pre-training synthetic corpus to date.

## 📢 What's New

- **[2026.08.20]** The [***Ultra-FineWeb-L1***](https://huggingface.co/datasets/openbmb/Ultra-FineWeb-L1) dataset is released! Built from Common Crawl snapshots, it undergoes main-text extraction, language filtering, heuristic filtering, sensitive-field replacement, customized cleaning, and deduplication, yielding **1T+ tokens (approximately 1.14 billion documents)**. We simultaneously release the L2 selected data [Ultra-FineWeb](https://huggingface.co/datasets/openbmb/Ultra-FineWeb/tree/main/data/ultrafineweb_l1_en_hq), selected by the [Ultra-FineWeb classifier](https://huggingface.co/openbmb/Ultra-FineWeb-classifier). To our best knowledge, this open-source web pre-training dataset covers the most recent Common Crawl snapshots, up to `CC-MAIN-2025-51`. 🚀🚀🚀
- **[2026.05.28]** The [***Ultra-FineWeb-L3***](https://huggingface.co/datasets/openbmb/Ultra-FineWeb-L3) dataset is released! The **L3 refined data** of **Ultra-FineWeb** via **Q&A Pair Generation** and **Multi-style Rewriting**, with **400B+ English** and **200B+ Chinese** tokens. To our best knowledge, it is the largest open-source Chinese pre-training synthetic corpus to date. 🚀🚀🚀
- **[2026.05.25]** ***[MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B) is released!***, the first model in the MiniCPM5 series. It is a dense 1B Transformer built for on-device, local deployment, and resource-constrained scenarios, reaching 1B-class open-source SOTA. Ultra-FineWeb serves as the core pre-training web dataset for MiniCPM5-1B. 
- **[2026.02.08]** The [***UltraData***](https://ultradata.openbmb.cn/) platform is now live, introducing the [L0-L4 tiered data management framework](https://arxiv.org/abs/2602.09003). ***Ultra-FineWeb*** serves as the **L2 selected layer** for general web data in this framework. 🔍🔍🔍
- **[2025.06.16]** The **Ultra-FineWeb-classifier** is now available on Hugging Face: [openbmb/Ultra-FineWeb-classifier](https://huggingface.co/openbmb/Ultra-FineWeb-classifier). 
- **[2025.06.06]** **Ultra-FineWeb-en** and **Ultra-FineWeb-zh** datasets are now available on Hugging Face, released alongside the [MiniCPM4 Series](https://huggingface.co/collections/openbmb/minicpm-4-6841ab29d180257e940baa9b) models.
- **[2025.05.15]** **Ultra-FineWeb** tops the Hugging Face Datasets Trending list, reaching the #1 spot! ⭐️⭐️⭐️
- **[2025.05.09]** **Ultra-FineWeb** technical report is available on [arXiv](https://arxiv.org/abs/2505.05427). 🔥🔥🔥

## 💡 Highlights

> **Abstract:** Data quality has become a key factor in enhancing model performance with the rapid development of large language models (LLMs). Model-driven data filtering has increasingly become a primary approach for acquiring high-quality data. However, it still faces two main challenges: (1) the lack of an efficient data verification strategy makes it difficult to provide timely feedback on data quality; and (2) the selection of seed data for training classifiers lacks clear criteria and relies heavily on human expertise, introducing a degree of subjectivity. To address the first challenge, we introduce an efficient verification strategy that enables rapid evaluation of the impact of data on LLM training with minimal computational cost. To tackle the second challenge, we build upon the assumption that high-quality seed data is beneficial for LLM training, and by integrating the proposed verification strategy, we optimize the selection of positive and negative samples and propose an efficient data filtering pipeline. This pipeline not only improves filtering efficiency, classifier quality, and robustness, but also significantly reduces experimental and inference costs. In addition, to efficiently filter high-quality data, we employ a lightweight classifier based on *fastText*, and successfully apply the filtering pipeline to two widely-used pre-training corpora, *FineWeb* and *Chinese FineWeb* datasets, resulting in the creation of the higher-quality ***Ultra-FineWeb*** dataset. ***Ultra-FineWeb*** contains approximately 1 trillion (T) English tokens and 120 billion (B) Chinese tokens. Empirical results demonstrate that the LLMs trained on Ultra-FineWeb exhibit significant performance improvements across multiple benchmark tasks, validating the effectiveness of our pipeline in enhancing both data quality and training efficiency.

<div align="center">
  <img src="assets/ultra-fineweb-pipeline.png" width="600"/>
</div>

- **Efficient Verification Strategy:** We propose a computationally efficient verification strategy that enables rapid evaluation of the impact of data on LLM training performance with minimal computational cost, significantly improving the efficiency of high-quality data filtering experiments.
- **Large-Scale High-Quality Pre-training Datasets:** We design and implement an efficient high-quality data filtering pipeline, applied to the FineWeb and Chinese FineWeb datasets, resulting in the creation of higher-quality datasets, which can facilitate high-quality LLM training.
- **Lightweight Classifier:** The Ultra-FineWeb classifier significantly reduces inference costs, achieving superior performance on extracted text from the same data source, thus validating the effectiveness of our proposed data filtering pipeline in enhancing data quality and training efficiency.

## 📈 Evaluation Results

We utilize the MiniCPM-1.2B model architecture with the MiniCPM3-4B tokenizer. Each experiment involves training on 100B tokens, allowing for comprehensive data performance validation within computationally efficient parameters. We employ Lighteval library for model evaluation, adopt 11 benchmarks to evaluate the performance of trained models, and all evaluation metrics are based on a zero-shot setting. The evaluation metrics include:

- **English benchmarks:** MMLU, ARC-C, ARC-E, CommonSenseQA, HellaSwag, OpenbookQA, PIQA, SIQA, and Winogrande.
- **Chinese benchmarks:** C-Eval and CMMLU.

Detailed evaluation results are reported below:

- **Individual data experiments.** We perform isolated training runs using single datasets, facilitating direct comparisons between differently processed data from identical sources.
<img src="assets/individual-english-table.png" alt="Individual English Table" width="75%">
<img src="assets/individual-chinese-table.png" alt="Individual Chinese Table" width="75%">
<img src="assets/individual-plot.png" alt="Individual Plot" width="100%">

- **Mixed Data Experiments.** We use a mix of 60% English data, 30% Chinese data, and 10% code data (StarCoder-v2).
<img src="assets/mix-table.png" alt="Mix Table" width="75%">
<img src="assets/mix-plot.png" alt="Mix Plot" width="100%">

- **Loss and Performance Estimation Results.** We use the performance estimation methods proposed in [Densing Law](https://arxiv.org/abs/2412.04315) for further analysis and verification of the effectiveness of Ultra-FineWeb.

<img src="assets/densing-law-table.png" alt="Densing Law Table" width="75%">
<img src="assets/densing-law-plot.png" alt="Densing Law Plot" width="100%">

## ❤️ Acknowledgements

- The ***Ultra-FineWeb classifier*** is built based on [fastText](https://fasttext.cc/). 
- The ***Ultra-FineWeb-en dataset*** is built based on [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb). 
- The ***Ultra-FineWeb-zh dataset*** is constructed based on [IndustryCorpus2](https://huggingface.co/datasets/BAAI/IndustryCorpus2), [MiChao](https://opendatalab.com/OpenDataLab/MiChao), [WuDao](https://data.baai.ac.cn/details/WuDaoCorporaText), [SkyPile](https://huggingface.co/datasets/Skywork/SkyPile-150B), [WanJuan](https://opendatalab.com/OpenDataLab/WanJuanCC), [ChineseWebText](https://huggingface.co/datasets/CASIA-LM/ChineseWebText2.0), [TeleChat](https://huggingface.co/datasets/Tele-AI/TeleChat-PTD), and [CCI3](https://huggingface.co/datasets/BAAI/CCI3-Data).

Thanks for their awesome work! Open-source contributions make Ultra-FineWeb possible! 🙌

## 🌟 Citation

If you find our work useful, please consider citing:

```bibtex
@misc{wang2025ultrafineweb,
  title={{Ultra-FineWeb}: Efficient Data Filtering and Verification for High-Quality LLM Training Data},
  author={Yudong Wang and Zixuan Fu and Jie Cai and Peijun Tang and Hongya Lyu and Yewei Fang and Zhi Zheng and Jie Zhou and Guoyang Zeng and Chaojun Xiao and Xu Han and Zhiyuan Liu},
  year={2025},
  eprint={2505.05427},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
}
```

And the main paper where Ultra-FineWeb is used:

```bibtex
@article{minicpm4,
  title={MiniCPM4: Ultra-Efficient LLMs on End Devices},
  author={MiniCPM Team},
  year={2025}
}
```

## 💳 License

This project is released under the [Apache 2.0](./LICENSE). Please note that since ***Ultra-FineWeb*** is built using multiple datasets, users should check the **LICENSE of each dataset individually** to ensure proper usage and compliance.