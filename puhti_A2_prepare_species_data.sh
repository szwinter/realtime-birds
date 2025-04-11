#!/bin/bash
#SBATCH --job-name=calculate_va
#SBATCH --account=project_2003104
#SBATCH --output=output/%A_%a
#SBATCH --ntasks=1 --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00 --partition=small
#SBATCH --array=0-258

ind=$SLURM_ARRAY_TASK_ID
N=${1:-100}

module load pytorch
hostname

srun python3 A2_prepare_species_data.py -n $N $ind
