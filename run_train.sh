#!/bin/bash
#SBATCH --job-name=rag-train
#SBATCH --output=slurmjobs/%j.out
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --tasks-per-node=1
#SBATCH --mem=64G
#SBATCH --time=5:59:00

scontrol show job -d $SLURM_JOBID | grep GRES
source activate rag-debias

# Command details:
#   --bias_dataset | allowed datasets: GenderBias-QA, PoliticBias-QA.
#   --lr | learning rates used: [3e-5, 1e-5].
#   --epochs | epochs used: [5, 10, 15]. 0 epochs only evaluates the model without training it.
#   --freeze_layers | numbers of layers used: [0, 1, 2, 3, 4].
#   --merge_weight | weights for merging used: [0.1, 0.3, 0.5, 0.7, 0.9].
#   --train_corpus | use '-' to merge training corpus parts (e.g., nq-msmarco-conclugen). allowed copora names: msmarco, fever, dbpedia, nq, hotpotqa, conclugen, argument, args_me
#   --save_dir | additionally include this argument with a directory to save the trained model, otherwise only the acc/bias evaluations of the model will be saved into a json file

accelerate launch embedder/train.py \
  --seed 0 \
  --model_type gte \
  --retriever_model thenlper/gte-base \
  --llm meta-llama/Llama-3.1-8B-Instruct \
  --judge gpt-4o-mini \
  --template basic \
  --pooling mean \
  --temperature 50.0 \
  --bias_dataset GenderBias-QA \
  --query_dataset rag-datasets/mini_wikipedia \
  --max_length 512 \
  --batch_size 8 \
  --embedding_batch 16 \
  --loss simcse \
  --optimizer adamw \
  --lr 3e-05 \
  --weight_decay 0.01 \
  --scheduler warmup_linear \
  --warmup_steps 10 \
  --epochs 5 \
  --freeze_layers 1 \
  --merge_weight 0.7 \
  --base_dataset_dir dataset/tasks/ \
  --base_index_dir dataset/filtered_bias_index/ \
  --train_corpus msmarco-fever-dbpedia \
  --eval_save_dir results/training \
  --cache_dir ${HUG_CACHE_DIR} \
  --save_prefix layer \
  --chunk_size 0 \
  --num_devices 1 \
  --use_wandb 0 \
  --wandb_project Debiasing-RAG \
  --wandb_name debiasing_rag_0_gte_thenlper_gte_base_meta_llama_Llama_3_1_8B_Instruct_gpt_4o_mini_basic_mean_50_0_GenderBias-QA_rag_datasets_mini_wikipedia_512_8_16_simcse_adamw_3e_05_0_01_warmup_linear_10_5_1_0_7 \
  --eval_bias 1 \
  --eval_overwrite 1 \
  --rag_type top1 \
  --bias_set train \
  --topk 3 \
  --top_p 0.0