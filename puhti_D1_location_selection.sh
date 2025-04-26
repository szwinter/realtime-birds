#!/bin/bash
#SBATCH --job-name=D1_location_selection
#SBATCH --account=project_2003104
#SBATCH --output=output/%A
#SBATCH --ntasks=1 --cpus-per-task=128
#SBATCH --mem=120G
#SBATCH --time=01:15:00 --partition=small

SPN=${1:-15}
AREA="${2:-all}"

mkdir -p output
module load pytorch
hostname

srun python3 D1_location_selection.py --spnumber=$SPN --area=$AREA
