#!/bin/bash
#SBATCH --job-name=B12_eval_realtime
#SBATCH --account=project_2003104
#SBATCH --output=output/%A_%a
#SBATCH --ntasks=1 --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=32:15:00 --partition=small
#SBATCH --array=0-262

IND=$SLURM_ARRAY_TASK_ID
STEP=1
D0=${1:-1}
D1=${2:-365}
MIGTRAIN=${3:-1}
SNP=${4:-0}
SPR=${5:-1}
SI=${6:-0}
F=${7:-10}
JN=${8:-4}


mkdir -p output
module load pytorch
module load hyperqueue
hostname

for i in {0..365}
do
if [[ $MIGTRAIN == 1 ]]; then
  M0=366
  M1=$((365+i*STEP))
  echo "Training migration model"
elif [[ $MIGTRAIN == 0 ]]; then
  M0=0
  M1=0
  echo "Not training migration model"
fi
S0=366
S1=$((365+i*STEP))
T0=$((365+i*STEP+1))
T1=$((365+i*STEP+7))
if (( $i % 1 )) ; then continue; fi
srun python3 B1_eval_species.py $IND --detstart $D0 --detstop $D1 --migstart $M0 --migstop $M1 \
  --spatstart $S0 --spatstop $S1 --teststart $T0 --teststop $T1 --savenewprior $SNP --savepred $SPR \
  --saveimages $SI  --factor $F --jn $JN
wait
done
