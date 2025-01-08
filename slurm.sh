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

INPUT_FILE="$1"
OUTPUT_FILE="$2"

# Check if input and output files are provided
if [ -z "$INPUT_FILE" ] || [ -z "$OUTPUT_FILE" ]; then
    echo "Usage: sbatch script.sh input.jsonl.zst output.jsonl"
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p slurm-logs

# Run the Python script using srun
srun python run.py "$INPUT_FILE" "$OUTPUT_FILE"