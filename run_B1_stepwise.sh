for i in {1..52}
do
    sbatch --array=0-3 puhti_B1_eval_species.sh 1 365 366 $((365+i*7)) 366 $((365+i*7)) $((365+i*7+1)) $((365+i*7+7)) 0 1
done
