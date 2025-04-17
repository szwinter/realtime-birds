# realtime-birds

# Structure of working directory at scratch

    ├── ...
    ├── data                      		# preprocessed data
    │   ├── species/          			# 
    │   │   ├── 000_acanthis_flammea/
    │   │   ├── 001_accipiter_gentilis/
    │   │   └── ...
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

Experiment 1
```console
sbatch --array=0-199 puhti_B11_eval_species.sh 1 365 1 365 1 365 366 730 transect app2023 1 1 1 1
sbatch --array=200-258 puhti_B11_eval_species.sh 1 365 1 365 1 365 366 730 transect app2023 1 1 1 1
species_id=0 detstart=1 detstop=365 migstart=1 migstop=365 spatstart=1 spatstop=365 teststart=366 teststop=730 priortype=transect namenewprior=app2023 savenewprior=1 saveimages=1 savepred=1 resetpriordet=0 resetpriormig=1 resetpriorspat=0 factor=10 jn=4
```

Experiment 2
```console
sbatch --array=0-199 puhti_B11_eval_species.sh 1 730 366 730 366 730 366 730 app2023 app20232024 0 1 1 0
sbatch --array=200-258 puhti_B11_eval_species.sh 1 730 366 730 366 730 366 730 app2023 app20232024 0 1 1 0
species_id=0 detstart=1 detstop=730 migstart=366 migstop=730 spatstart=366 spatstop=730 teststart=366 teststop=730 priortype=app2023 namenewprior=app20232024 savenewprior=0 saveimages=1 savepred=1 resetpriordet=0 resetpriormig=0 resetpriorspat=0 factor=10 jn=4
```

Experiment 3
```console
sbatch --array=0-199 puhti_B12_eval_realtime.sh 7 1 365 1 app2023 app2024rt 0 1 1
sbatch --array=200-258 puhti_B12_eval_realtime.sh 7 1 365 1 app2023 app2024rt 0 1 1
species_id=0 detstart=1 detstop=365 migstart=366 migstop=365 spatstart=366 spatstop=365 teststart=366 teststop=372 priortype=app2023 namenewprior=app2024rt_365 savenewprior=0 saveimages=1 savepred=1 resetpriordet=0 resetpriormig=0 resetpriorspat=0 factor=10 jn=4
species_id=0 detstart=1 detstop=365 migstart=366 migstop=372 spatstart=366 spatstop=372 teststart=373 teststop=379 priortype=app2023 namenewprior=app2024rt_372 savenewprior=0 saveimages=1 savepred=1 resetpriordet=0 resetpriormig=0 resetpriorspat=0 factor=10 jn=4
.....
```

## Various relevant shell commands

Send maps to Allas for public sharing
```console
find . -name '*.jpeg' | cpio -pdm  ../export/
find data/species/ -type f -name *app2023.tif | xargs -L 1 a-flip
```
