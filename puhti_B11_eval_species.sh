#!/bin/bash
#SBATCH --job-name=B11_eval_species
#SBATCH --account=project_2003104
#SBATCH --output=output/%A_%a
#SBATCH --ntasks=1 --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=00:59:00 --partition=small
#SBATCH --array=0-262

IND=$SLURM_ARRAY_TASK_ID
D0=${1:-1}
D1=${2:-365}
M0=${3:-366}
M1=${4:-730}
S0=${5:-366}
S1=${6:-730}
T0=${7:-366}
T1=${8:-730}
PT="${9:-transect}"
NNP="${10:-app}"
SNP=${11:-0}
SPR=${12:-0}
SI=${13:-1}
RPM=${14:-1}
F=${15:-10}
JN=${16:-4}

mkdir -p output
module load pytorch
hostname

srun python3 B1_eval_species.py $IND --detstart $D0 --detstop $D1 --migstart $M0 --migstop $M1 \
  --spatstart $S0 --spatstop $S1 --teststart $T0 --teststop $T1 \
  --priortype $PT --namenewprior $NNP \
  --savenewprior $SNP --savepred $SPR \
  --saveimages $SI --resetpriormig $RPM --factor $F --jn $JN
