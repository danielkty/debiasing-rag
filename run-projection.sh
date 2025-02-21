#!/bin/bash
#SBATCH --job-name=rag-projection   
#SBATCH --output=slurmjobs/%j.out
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --time=11:59:00
#SBATCH --mem=64G

scontrol show job -d $SLURM_JOBID | grep GRES
source activate rag-debias
python src/rag-projection.py -query_d GenderBias-QA -corpus_d mteb/nq -llm meta-llama/Llama-3.1-8B-Instruct -p_word female
echo "job finished";