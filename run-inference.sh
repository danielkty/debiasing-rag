#!/bin/bash
#SBATCH --job-name=rag-inference
#SBATCH --output=slurmjobs/%j.out
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --tasks-per-node=1
#SBATCH --time=1:59:00
#SBATCH --mem=64G

scontrol show job -d $SLURM_JOBID | grep GRES
source activate rag-debias
python src/rag-inference.py -query_d PoliticBias-QA -corpus_d Pol_NLI -llm meta-llama/Llama-3.1-8B-Instruct -rag 1 -set test -rm thenlper/gte-base -judge gpt-4o-mini
echo "job finished";