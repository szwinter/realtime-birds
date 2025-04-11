# realtime-birds

# Structure of working directory at scratch

    ├── ...
    ├── data                      		# preprocessed data
    │   ├── species/          			# 
    │   ├── XData.pickle	       		# 
    │   └── migration_prior_params.pickle       # 
    ├── orig_data				# original data after unzipping 
    │   ├── a_maps/				# 
    │   ├── b_maps/				# 
    │   ├── va_maps/				# 
    │   ├── prior predictions/			# 
    │   └── meta.RData     			# 
    └── results

# Pipleline in Mahti

While the queue waiting times in Mahti are generally less that in Puhti, it limits the number of batch jobs to 200. Thus, we need to run each command effectively twice with different array job index ranges.

```console
sbatch --array=0-199 puhti_B11_eval_species.sh 1 365 1 365 1 365 366 730 transect app2023 1 1 1 1
sbatch --array=200-258 puhti_B11_eval_species.sh 1 365 1 365 1 365 366 730 transect app2023 1 1 1 1

sbatch --array=0-199 puhti_B11_eval_species.sh 1 730 366 730 366 730 366 730 app2023 app20232024 0 1 1 0
sbatch --array=200-258 puhti_B11_eval_species.sh 1 730 366 730 366 730 366 730 app2023 app20232024 0 1 1 0

sbatch --array=0-199 puhti_B12_eval_realtime.sh 7 1 365 1 app2023 app2024rt 0 1 1
sbatch --array=200-258 puhti_B12_eval_realtime.sh 7 1 365 1 app2023 app2024rt 0 1 1

```