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

## All data till now analysis
### Step 1, year 2023
Load spatial transect prior. Fit detection using 2023, update migration from prior using 2023, update spatial using 2023. Save spatial.
```console
sbatch puhti_B11_eval_species.sh 1 365 1 365 1 365 366 731 transect app23 1 1 1 1
species_id=0 detstart=1 detstop=365 migstart=1 migstop=365 spatstart=1 spatstop=365 teststart=366 teststop=731 priortype=transect namenewprior=app23 savenewprior=1 saveimages=1 savepred=1 resetpriordet=0 resetpriormig=1 resetpriorspat=0 factor=10 jn=4
```

### Step 2, year 2024
Load 2023 spatial posterior. Fit detection using 2023 + 2024, update migration from prior using 2024, update spatial using 2024. Save spatial.
```console
sbatch puhti_B11_eval_species.sh 1 731 366 731 366 731 366 731 app23 app2324 1 1 1 0
species_id=0 detstart=1 detstop=731 migstart=366 migstop=731 spatstart=366 spatstop=731 teststart=366 teststop=731 priortype=app23 namenewprior=app2324 savenewprior=1 saveimages=1 savepred=1 resetpriordet=0 resetpriormig=0 resetpriorspat=0 factor=10 jn=4
```

### Step 3, year 2025
Load 2024 spatial posterior. Fit detection using 2023 + 2024, update migration from prior using 2025, update spatial using 2025. Save spatial.
```console
sbatch puhti_B11_eval_species.sh 1 731 732 1096 732 1096 732 1096 app2324 app232425 1 1 1 0
species_id=0 detstart=1 detstop=731 migstart=732 migstop=1096 spatstart=732 spatstop=1096 teststart=732 teststop=1096 priortype=app2324 namenewprior=app232425 savenewprior=1 saveimages=1 savepred=1 resetpriordet=0 resetpriormig=0 resetpriorspat=0 factor=10 jn=4
```

### Experiment realtime 2024
Note that detection is set to the first year. Is this desired?
```console
sbatch puhti_B12_eval_realtime.sh 7 1 365 1 app23 app24rt 0 1 0
species_id=0 detstart=1 detstop=365 migstart=366 migstop=365 spatstart=366 spatstop=365 teststart=366 teststop=372 priortype=app23 namenewprior=app24rt_365 savenewprior=0 saveimages=0 savepred=1 resetpriordet=0 resetpriormig=0 resetpriorspat=0 factor=10 jn=4
species_id=0 detstart=1 detstop=365 migstart=366 migstop=372 spatstart=366 spatstop=372 teststart=373 teststop=379 priortype=app23 namenewprior=app24rt_372 savenewprior=0 saveimages=0 savepred=1 resetpriordet=0 resetpriormig=0 resetpriorspat=0 factor=10 jn=4
.....
```

### Location selection
```console
sbatch puhti_D1_location_selection.sh 80 all
sbatch puhti_D1_location_selection.sh 197 north
sbatch puhti_D1_location_selection.sh 197 south
```


## Various relevant shell commands

Send maps to Allas for public sharing
```console
find data/species/ -type f -name *app2023.tif | xargs -L 1 a-flip
```

Move all jpeg images from subdirectories to respective subdirectories in another directory
```console
find . -name '*.jpeg' | cpio -pdm  ../export/
```

