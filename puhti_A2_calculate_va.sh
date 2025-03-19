#!/bin/bash
#SBATCH --job-name=calculate_va
#SBATCH --account=project_2003104
#SBATCH --output=output/%A_%a
#SBATCH --ntasks=1 --cpus-per-task=2
#SBATCH --mem-per-cpu=16G
#SBATCH --time=00:30:00 --partition=small
#SBATCH --array=0-262

ind=$SLURM_ARRAY_TASK_ID
N=${1:-100}

module load pytorch
hostname

srun python3 A1_calculate_va.py -n $N $ind
