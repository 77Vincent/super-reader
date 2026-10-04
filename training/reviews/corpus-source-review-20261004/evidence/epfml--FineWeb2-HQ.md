---
task_categories:
- text-generation
language:
- ru
- zh
- de
- ja
- es
- fr
- it
- pt
- pl
- nl
- id
- tr
- cs
- vi
- sv
- fa
- ar
- el
- da
- hu
pretty_name: FineWeb2-HQ
configs:
- config_name: rus_Cyrl
  data_files:
  - split: train
    path: rus_Cyrl/*
- config_name: cmn_Hani
  data_files:
  - split: train
    path: cmn_Hani/*
- config_name: deu_Latn
  data_files:
  - split: train
    path: deu_Latn/*
- config_name: jpn_Jpan
  data_files:
  - split: train
    path: jpn_Jpan/*
- config_name: spa_Latn
  data_files:
  - split: train
    path: spa_Latn/*
- config_name: fra_Latn
  data_files:
  - split: train
    path: fra_Latn/*
- config_name: ita_Latn
  data_files:
  - split: train
    path: ita_Latn/*
- config_name: por_Latn
  data_files:
  - split: train
    path: por_Latn/*
- config_name: pol_Latn
  data_files:
  - split: train
    path: pol_Latn/*
- config_name: nld_Latn
  data_files:
  - split: train
    path: nld_Latn/*
- config_name: ind_Latn
  data_files:
  - split: train
    path: ind_Latn/*
- config_name: tur_Latn
  data_files:
  - split: train
    path: tur_Latn/*
- config_name: ces_Latn
  data_files:
  - split: train
    path: ces_Latn/*
- config_name: vie_Latn
  data_files:
  - split: train
    path: vie_Latn/*
- config_name: swe_Latn
  data_files:
  - split: train
    path: swe_Latn/*
- config_name: fas_Arab
  data_files:
  - split: train
    path: fas_Arab/*
- config_name: arb_Arab
  data_files:
  - split: train
    path: arb_Arab/*
- config_name: ell_Grek
  data_files:
  - split: train
    path: ell_Grek/*
- config_name: dan_Latn
  data_files:
  - split: train
    path: dan_Latn/*
- config_name: hun_Latn
  data_files:
  - split: train
    path: hun_Latn/*
size_categories:
- 100M<n<1B
license: odc-by
---
# FineWeb2-HQ

## Dataset summary

FineWeb2-HQ is a **high-quality, model-filtered pretraining dataset** derived as a subset of [**FineWeb2**](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2), spanning **20 languages**. It enables around 6x faster pretraining compared to the base dataset. FineWeb2-HQ was created by selecting the **top 10% quality documents of FineWeb2** in each language, based on scores assigned by a deep learning classifier trained to identify **structured and knowledge-rich samples** using [**XLM-RoBERTa**](https://huggingface.co/FacebookAI/xlm-roberta-base) **embeddings**.

<center>
  <img src="https://huggingface.co/datasets/epfml/FineWeb2-HQ/raw/main/agg_score_plot.svg" style="width: 70%;" />
</center>

Validation was performed by pretraining **1B-parameter LLM models** (llama-like architecture) across multiple languages and writing systems (scripts). Evaluations on **CMMLU (Chinese) and MMLU (German & French)** demonstrate that **FineWeb2-HQ matches FineWeb2 performance when trained with 6x fewer tokens, and outperforms it when fully trained**. Additionally, **improvements were observed across other benchmarks**, such as outperforming its English cousins [DCLM](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0-parquet) and [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu).

For more details, see our paper [Enhancing Multilingual LLM Pretraining with Model-Based Data Selection](https://arxiv.org/abs/2502.10361).

## Key features

- **High-quality selection**: Top 10% of FineWeb2 documents by quality
- **Multilingual coverage**: 20 languages, ensuring diverse linguistic representation
- **Model-based filtering**: Uses an XLM-RoBERTa embedding-based classifier to score documents
- **Enhanced benchmark performance**: Surpasses FineWeb2 benchmark performance
- **Fully open**: Emphasis on transparency

## Languages and subsets

|Subset name|Language name|Number of documents|Disk size|
|----------|-----------------|------------:|----------:|
| rus_Cyrl | Russian         |  55,220,956 | 1.2T      |
| cmn_Hani | Chinese         |  54,211,986 | 784G      |
| deu_Latn | German          |  43,095,728 | 618G      |
| spa_Latn | Spanish         |  40,057,637 | 515G      |
| jpn_Jpan | Japanese        |  34,185,427 | 393G      |
| fra_Latn | French          |  32,248,772 | 483G      |
| ita_Latn | Italian         |  21,180,304 | 269G      |
| por_Latn | Portuguese      |  18,135,468 | 222G      |
| pol_Latn | Polish          |  13,384,885 | 168G      |
| nld_Latn | Dutch           |  12,920,963 | 160G      |
| ind_Latn | Indonesian      |   8,911,149 | 125G      |
| tur_Latn | Turkish         |   8,578,808 | 100G      |
| ces_Latn | Czech           |   5,995,459 | 104G      |
| arb_Arab | Arabic          |   5,560,599 | 94G       |
| fas_Arab | Persian         |   5,107,187 | 69G       |
| hun_Latn | Hungarian       |   4,527,332 | 79G       |
| swe_Latn | Swedish         |   4,382,454 | 61G       |
| ell_Grek | Greek           |   4,346,440 | 84G       |
| dan_Latn | Danish          |   4,082,751 | 61G       |
| vie_Latn | Vietnamese      |   4,003,956 | 59G       |

The approach as described in the paper is easy to extend to other languages as well, and we might consider adding new languages to an upcoming version of the present dataset.

We also separately release the computed general-purpose embedding vectors for the the full sets of the original FineWeb2 dataset (not just the HQ subsets), in the respective languages, as they can be useful for other applications beyond quality filtering: [FineWeb2-embedded](https://huggingface.co/datasets/epfml/FineWeb2-embedded).

## Dataset structure

### Data fields

Each data entry includes the original [FineWeb2 data fields](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2#data-fields) with the addition of:
- `quality_score`: quality score obtained by the quality classifier
- `embeddings`: array of float arrays containing 768-dimensional XLM-RoBERTa embeddings for every 512 token chunk of the tokenized text

### Data instance

```json
{
  "id": "<urn:uuid:f26003c7-6084-4791-b3fe-240eedc37e76>",
  "text": "Plutonium ist einer der gefährlichsten Stoffe der Welt. Es entsteht als hochgiftiges und radioaktives Nebenprodukt der Energiegewinnung in Atomkraftwerken. Wer nur ein Millionstel Gramm – ein kaum staubkorngroßes Teilchen – der Substanz einatmet, kann daran sterben. In der Natur kommt der Stoff nur in geringsten Mengen vor, wird aber künstlich hergestellt, weil man damit Bomben bauen kann. Je nach Reinheitsgrad reichen für eine Atombombe bereits fünf Kilogramm. Bis zum Beginn der achtziger Jahre des letzten Jahrhunderts hatten die Reaktoren weltweit bereits rund 300.000 Kilogramm erbrütet. Jährlich kommen etwa 20.000 Kilo hinzu. Genau dieser Stoff wird zu Land und zu Wasser um den ganzen Erdball herum transportiert. Legendär sind die Castor-Transporte, bei denen unter strengsten Sicherheitsvorkehrungen und entsprechenden Kosten abgebrannte Brennelemente aus deutschen Kernkraftwerken zur Wiederaufbereitung nach La Hague (Frankreich) oder Sellafield (Großbritannien) gebracht werden. Erst vergangenen Mai hat ein Frachter die größte Menge wiederaufbereiteten Mülls aller Zeiten von Frankreich nach Japan gebracht. Nicht auszudenken, was ein Unfall auf See bedeuten würde.",
  "date": "2014-03-16T08:53:38Z",
  "dump": "CC-MAIN-2014-10",
  "embeddings": [[ ... ]],
  "file_path": "s3://commoncrawl/crawl-data/CC-MAIN-2014-10/segments/1394678702159/warc/CC-MAIN-20140313024502-00039-ip-10-183-142-35.ec2.internal.warc.gz",
  "language": "deu",
  "language_score":  0.9983288645744324,
  "language_script": "Latn",
  "minhash_cluster_size": 2,
  "top_langs": {"deu_Latn_score": 0.9983288645744324},
  "url": "http://www.greenpeace.org/austria/de/themen/atom/probleme/atomtransporte/",
  "quality_score": 0.06472613662481308
}
```

## Usage

You can load the dataset in Python using `datasets`:

```python
from datasets import load_dataset

dataset = load_dataset("epfml/FineWeb2-HQ", "deu_Latn")
```

## Licensing information

Like FineWeb2, this dataset is released under [Open Data Commons Attribution License (ODC-By) v1.0](https://opendatacommons.org/licenses/by/1-0/) license and is subject to [CommonCrawl's Terms of Use](https://commoncrawl.org/terms-of-use).

## Dataset origin
Being a subset of FineWeb2, this data covers websites over the 2013-2024 time period.

FineWeb2 is sourced from the internet at large, it is very likely that some personable identifiable information (PII) will be present, even if the FineWeb2 processing has already anonymized email addresses and public IP addresses. If you find your own PII and would like it removed, please fill out the [FineWeb2 PII removal/opt out form](https://forms.gle/VyNT3ZAUPZjPuWp39).

CommonCrawl respects robots.txt at crawl time, but if you are a webmaster and find your website in FineWeb2 and would like to have it removed, you may also use the [FineWeb2 PII removal/opt out form](https://forms.gle/VyNT3ZAUPZjPuWp39).

## Considerations for Using the Data
Before using this dataset for training models, we recommend performing additional filtering for sensitive content such as PII or harmful content.
For the aspects of social impact, discussion of biases, and known limitations, we also refer to the [FineWeb2 documentation](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2).

## Citation information
If you use this dataset in your research or applications, please use the following citation:
```
@article{messmer2025multilingdatacomp,
  title={Enhancing Multilingual LLM Pretraining with Model-Based Data Selection}, 
  author={Bettina Messmer and Vinko Sabolčec and Martin Jaggi},
  journal={arXiv},
  year={2025},
  url={https://arxiv.org/abs/2502.10361}, 
}
```