#!/bin/bash

#SBATCH --job-name=register-labeling-simple
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --output=slurm-logs/%j.out
#SBATCH --error=slurm-logs/%j.err
#SBATCH --account=project_2011770
#SBATCH --partition=gpusmall

module use /appl/local/csc/modulefiles
module load pytorch/2.4

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 PACKAGE" >&2
    echo >&2
    echo "example: $0 fin_Latn.shuf.zst" >&2
    exit 1
fi

PACKAGE="$1"

module use /appl/local/csc/modulefiles
module load pytorch/2.4
