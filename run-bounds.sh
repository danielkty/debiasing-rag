#!/bin/bash
#SBATCH --job-name=rag-get-bounds
#SBATCH --output=slurmjobs/%j.out
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --tasks-per-node=1
#SBATCH --mem=64G
#SBATCH --time=5:59:00

scontrol show job -d $SLURM_JOBID | grep GRES
source activate rag-debias
python src/rag-get-bound.py -query_d PoliticBias-QA -corpus_d Pol_NLI -split train -k 3
echo "job finished";
