#!/bin/bash
#SBATCH --job-name=daily_update_prototype
#SBATCH --account=project_2003104
#SBATCH --output=output/%A
#SBATCH --partition=small
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

COUNT=${1:-0}

mkdir -p output
module load tensorflow/2.18
hostname

srun python3 hello.py $COUNT

COUNT=$((COUNT+1))
if (( COUNT < 10 )); then
	startTime=$(date -d "16:20:00 2025-05-13 +$COUNT min" +%Y-%m-%dT%H:%M:%S)
	echo $startTime
	sbatch --begin=$startTime Z1_self_repeat.sh $COUNT
fi
