#!/bin/bash
#SBATCH --job-name=B1_eval_species
#SBATCH --account=project_2003104
#SBATCH --output=output/%A_%a
#SBATCH --ntasks=1 --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=22:15:00 --partition=small
#SBATCH --array=40-262

IND=$SLURM_ARRAY_TASK_ID
STEP=1
D0=${1:-1}
D1=${2:-365}
SNP=${3:-0}
SPR=${4:-1}
SI=${5:-1}
F=${6:-10}
JN=${7:-4}


mkdir -p output
module load pytorch
module load hyperqueue
hostname

for i in {1..365}
do
M0=366
M1=$((365+i*STEP))
S0=366
S1=$((365+i*STEP))
T0=$((365+i*STEP+1))
T1=$((365+i*STEP+7))
if ! (( $i % 7 )) ; then continue; fi
srun python3 B1_eval_species.py $IND --detstart $D0 --detstop $D1 --migstart $M0 --migstop $M1 \
  --spatstart $S0 --spatstop $S1 --teststart $T0 --teststop $T1 --savenewprior $SNP --savepred $SPR \
  --saveimages $SI  --factor $F --jn $JN
wait
done
