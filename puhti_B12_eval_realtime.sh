#!/bin/bash
#SBATCH --job-name=B12_eval_realtime
#SBATCH --account=project_2003104
#SBATCH --output=output/%A_%a
#SBATCH --ntasks=1 --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=15:15:00 --partition=small
#SBATCH --array=0-196

IND=$SLURM_ARRAY_TASK_ID
STEP=${1:-7}
D0=${2:-1}
D1=${3:-365}
MIGTRAIN=${4:-1}
PT="${5:-transect}"
NNPPREFIX="${6:-app}"
SNP=${7:-0}
SPR=${8:-1}
SI=${9:-0}
F=${10:-10}
JN=${11:-4}

mkdir -p output
module load pytorch
hostname
STEPN=$((365 / STEP))

for (( i=0; i<=STEPN; i++ ))
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
    NNP=$(printf "%s_%d" $NNPPREFIX $S1)
    if (( $i % 1 )) ; then continue; fi # to skip some steps
    srun python3 B1_eval_species.py $IND --detstart $D0 --detstop $D1 --migstart $M0 --migstop $M1 \
      --spatstart $S0 --spatstop $S1 --teststart $T0 --teststop $T1 \
      --priortype $PT --namenewprior $NNP --savenewprior $SNP --savepred $SPR --saveimages $SI \
      --factor $F --jn $JN
    wait
done
