# 🔍 Mitigating Bias in RAG: Controlling the Embedder

This is the repository for **Mitigating Bias in RAG: Controlling the Embedder**! This repository includes the GenderBias-QA and PoliticBias-QA datasets, as well as the training and inference code to reproduce our results. The goal of this repo is to bias the embedder for these datasets to achieve an unbiased RAG system.

---

## Table of Contents

- [Getting Started](#getting-started)
- [Sorting Documents](#sorting-documents)
- [Fine-tuning an Embedder](#fine-tuning)
- [Evaluation and Inferencing](#evaluation-and-inferencing)
- [Evaluating Projections](#evaluating-projections)

---

## Getting Started

This repository includes:
- **Datasets:** GenderBias-QA and PoliticBias-QA
- **Training Script:** Fine-tune an embedder with PEFT or WiSE-FT to adjust its bias
- **Evaluation Script:** Evaluate the performance of the biased embedder within a full RAG pipeline
- **Projection Script:** Assess projections in the embedding space for biasing retrieved documents

First, install the Python packages by running:
```bash
conda env create --file environment.yml
```

Second, enter your API keys and cache directory in env_config.sh and run:
```bash
source env_config.sh
```

## Sorting Documents

To prepare for contrastive fine-tuning, you should first sort positive and negative documents from a corpus. To do this, run:
```bash
bash run-bounds.sh
```

This command runs `src/rag-get-bound.py` which will save the indices of the positive and negative documents in dataset/filtered_bias_index. The saved indices will later be used during fine-tuning.

Here are some of the main arguments to be passed into `rag-get-bound.py`:
- `-query_d`: Chooses the bias dataset (either `PoliticBias-QA` or `GenderBias-QA`).
- `-corpus_d`: Specifies the corpus in which the documents get sorted from. Possible choices include:
  - `mteb/msmarco`
  - `mteb/dbpedia`
  - `mteb/fever`
  - `mteb/nq`
  - `mteb/hotpotqa`
  - `webis/args_me`
  - `webis-argument-framing`
  - `webis/conclugen`
  - `Pol_NLI`
  
### Fine-tuning

After you have sorted your documents for training, you can fine-tune an embedder to change its bias by running:
```bash
bash run-train.sh
```

This command runs `embedder/train.py` to train the embedder and evaluate both its accuracy and bias. The evaluation results are saved to a JSON file. By default, the trained model is not saved unless you provide the `--save_dir` argument.

Here are some of the main arguments to be passed into `train.py`:
- `--bias_dataset`: `GenderBias-QA` or `PoliticBias-QA`.
- `--lr`: Learning rates to use (e.g., [3e-5, 1e-5]).
- `--epochs`: Number of epochs (e.g., [5, 10, 15]). Setting this to 0 will evaluate the model without training.
- `--freeze_layers`: Number of layers to freeze (e.g., [0, 1, 2, 3, 4]).
- `--merge_weight`: Weights for merging (e.g., [0.1, 0.3, 0.5, 0.7, 0.9]).
- `--train_corpus`: Specify training corpus parts by using a hyphen to merge names (e.g., `nq-msmarco-conclugen`). Allowed corpus names include: msmarco, fever, dbpedia, nq, hotpotqa, conclugen, argument, args_me.
- `--save_dir`: Optionally provide a directory to save the trained model; if omitted, only the accuracy and bias evaluations will be saved to a JSON file.


### Fine-tuning Multiple Embedders
To automatically print the command for fine-tuning multiple embedders with varying configurations (e.g., learning rate, epochs, numbers of frozen layers), modify `experiments/make_commands.py` and run:
```bash
cd experiments
python make_commands.py
```

This will create a text file with each line being a command to run one fine-tuning job.


### Evaluation and Inferencing
After an embedder has been trained, you may test the performance of the embedder within a full rag pipeline. This will evaluate the bias of the embedder and RAG pipeline. To evaluate, run:
```bash
bash run-inference.sh
```

This command runs `src/rag-inference.py` which will evaluate the bias of a full RAG pipeline. You will need to provide the LLM, corpus, and embedder for evaluation.

Here are some of the main arguments to be passed into `rag-inference.py`:
- `-query_d`: Specifies the queries to retrieve documents for. Can be chosen between `GenderBias-QA` or `PoliticBias-QA`.
- `-corpus_d`: Specifies the corpus for inference. Can be chosen between `mteb/nq`, `mteb/hotpotqa`, or `Pol_NLI`.
- `-llm`: Specifies the Huggingface LLM to inference on
- `-rm`: Specifies the embedder to retrieve documents with. Select a base embedder from Huggingface or a directory to a trained embedder.

### Evaluating Projections
You can also bias embedders by projecting them in the embedding space instead of fine-tuning. To do so, run:
```bash
bash run-projection.sh
```

This command runs `src/rag-projection.py` which will evaluate the bias of the full RAG pipeline using projections in a base embedder (`gte-base` by default). 

Here are some of the main arguments to be passed into `rag-projection.py`:
- `-query_d`: Specifies the queries to retrieve documents for. Can be chosen between `GenderBias-QA` or `PoliticBias-QA`.
- `-corpus_d`: Specifies the corpus for inference. Can be chosen between `mteb/nq`, `mteb/hotpotqa`, or `Pol_NLI`.
- `-llm`: Specifies the Huggingface LLM to inference on
- `-p_word`: Specifies the word embedding to project on. Should be `female` for `GenderBias-QA` and `republican` for `PoliticBias-QA`
