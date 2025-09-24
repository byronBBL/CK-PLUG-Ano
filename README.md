Parameters vs. Context: Fine-Grained Control of Knowledge Reliance in Language Models

Code for the paper "Parameters vs. Context: Fine-Grained Control of Knowledge Reliance in Language Models"

## Overview

![Parameters vs. Context](overview.png)

Retrieval-Augmented Generation (RAG) mitigates hallucinations in Large Language Models (LLMs) by integrating external knowledge.
However, conflicts between parametric knowledge and retrieved context pose challenges, particularly when retrieved information is unreliable or the model's internal knowledge is outdated. 
In such cases, LLMs struggle to determine whether to rely more on their own parameters or the conflicted context.
To address this, we propose **CK-PLUG**, a plug-and-play method for controlling LLMs' reliance on parametric and contextual knowledge. 
We introduce a novel knowledge consistency metric, ***Confidence Gain***, which detects knowledge conflicts by measuring entropy shifts in token probability distributions after context insertion.
CK-PLUG then enables fine-grained control over knowledge preference by adjusting the probability distribution of tokens with negative confidence gain through a single tuning parameter.
Experiments demonstrate CK-PLUG's ability to significantly regulate knowledge reliance in counterfactual RAG scenarios while maintaining generation fluency and knowledge accuracy.
For instance, on LLaMA-3-8B, memory recall (MR) of RAG response can be adjusted within a broad range (9.9\%-71.9\%), compared to the baseline of 42.1\%.
Moreover, CK-PLUG supports adaptive control based on the model's confidence in both internal and external knowledge, achieving consistent performance improvements across various general RAG tasks. 

![CK-PLUG](framework.png)

## Datasets

### Knowledge Reliance Controlling

Download the datasets for `./kr_data`:

NQ: [Google Drive](https://drive.google.com/file/d/1DJ1ajmLNAKVTBWnM7SkP93EYQ2cav3Mk/view)
ConFiQA: [GitHub Repository](https://github.com/byronBBL/Context-DPO/tree/master/ConFiQA)
MQuAKE: [GitHub Repository](https://github.com/princeton-nlp/MQuAKE/blob/main/datasets/MQuAKE-CF-3k-v2.json)

### RAG Downstream Tasks

In order to obtain the retrieved passages, we provide the implementation code of the retrieval stage at [\retrieval](https://github.com/byronBBL/CK-PLUG/blob/master/retrieval). Specifically,
- [retrievers.py](https://github.com/byronBBL/CK-PLUG/blob/master/retrieval/retrievers.py): class definition of retrievers
- [retrieval.py](https://github.com/byronBBL/CK-PLUG/blob/master/retrieval/retrieval.py): retrieval pipeline (referenced from [beir](https://github.com/beir-cellar/beir))
- [preprocess.py](https://github.com/byronBBL/CK-PLUG/blob/master/retrieval/preprocess.py): including data preprocessing operations such as downloading, format alignment, sampling, etc.
- [main.py](https://github.com/byronBBL/CK-PLUG/blob/master/retrieval/main.py): entry file


Using the segmented corpus, we can run the following command to perform the retrieval operation:
```
python \retrieval\main.py --retriever bge --corpus_path wikipedia_100_2019_08_01.jsonl --topk 20
```
Remember to modify the file path for your environment.

## Experiments

Setup with transformers (incorporating CK-PLUG)

```bash
pip install -e transformers-4.49
```

Run the **knowledge control evaluation** on the NQ, ConFiQA and MQuAKE using the following command:  

```bash
python eval_NQ.py --model_name ./model_path --mode ck
python eval_ConFiQA.py --model_name ./model_path --mode ck
python eval_MQuAKE.py --model_name ./model_path --mode ck
```

Run the **adaptive enhancement evaluation** on the [KILT](https://huggingface.co/datasets/facebook/kilt_tasks/viewer/hotpotqa/validation) using the following command:  

```bash
python eval_rag.py --model_name ./model_path --mode ck --adaptive True --input_file rag_data --task rag_task
```
