#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 00:16:51 2025

@author: gtikhono
"""

#!/bin/bash
#SBATCH --job-name=B1_eval_species
#SBATCH --account=project_2003104
#SBATCH --output=output/%A_%a
#SBATCH --ntasks=1 --cpus-per-task=4
#SBATCH --mem-per-cpu=20G
#SBATCH --time=00:59:00 --partition=small
#SBATCH --array=0-3

IND=$SLURM_ARRAY_TASK_ID
D0=${1:-1}
D1=${2:-365}
M0=${3:-366}
M1=${4:-730}
S0=${5:-366}
S1=${6:-730}
T0=${7:-366}
T1=${8:-730}
SNP=${9:-0}
SI=${10:-1}
F=${11:-10}
JN==${12:-4}


module load pytorch
hostname

srun python3 A1_calculate_va.py $IND --detstart $D0 --detstop $D1 --migstart $M0 --migstop $M1 \
  --spatstart $S0 --spatstop $S1 --teststart $T0 --teststop $T1 --savenewprior $SNP --saveimages $SI \
  --factor $F --jn $JN
