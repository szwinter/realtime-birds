# %% Imports
import os
import boto3
import tqdm
import pickle
import numpy as np
import argparse
from matplotlib import pyplot as plt
import pandas as pd

path_project = "/scratch/project_2003104/gtikhono/realtime_birds"
dir_results = "results"
df_sp_model = pd.read_csv(os.path.join(path_project, "data", "modeled_species.csv"))
sp_list = list(df_sp_model.species)
factor = 10


# %%
os.environ["AWS_REQUEST_CHECKSUM_CALCULATION"] = "when_required"
os.environ["AWS_RESPONSE_CHECKSUM_VALIDATION"] = "when_required"
s3_resource = boto3.resource('s3', endpoint_url='https://a3s.fi')
bucket_name = "gtikhono-MKapp-paper-export"
s3_resource.create_bucket(Bucket=bucket_name, ACL="public-read")
my_bucket = s3_resource.Bucket(bucket_name)
for bucket in s3_resource.buckets.all():
    print(bucket.name)

# %%
df_sp_model["migration"] = False
for j, sp in enumerate(tqdm.tqdm(sp_list)):
    path_sp = os.path.join(path_project, dir_results, sp)
    path_migration_parameters_24 = os.path.join(path_sp, sp + "_migration_app23_det(1_731)_mig(366_731)_dyn(366_731)_test000(366_731).csv")
    path_migration_parameters_25 = os.path.join(path_sp, sp + "_migration_app2324_det(1_731)_mig(732_1096)_dyn(732_1096)_test000(732_1096).csv")
    if os.path.exists(path_migration_parameters_24) and os.path.exists(path_migration_parameters_25):
        df_sp_model.loc[j, "migration"] = True
        s3_resource.Object(bucket_name, f'migration_parameters/{sp + "_migration_24.csv"}').upload_file(path_migration_parameters_24, ExtraArgs={'ACL': 'public-read'})
        s3_resource.Object(bucket_name, f'migration_parameters/{sp + "_migration_25.csv"}').upload_file(path_migration_parameters_25, ExtraArgs={'ACL': 'public-read'})
    else:
        print("Missing migration parameter files for %d-%s" % (j,sp))
    
path_df_sp_model_migration = os.path.join(path_project, "postprocessed", "modeled_species_migration.csv")
df_sp_model.to_csv(path_df_sp_model_migration, index=False)
s3_resource.Object(bucket_name, "modeled_species_migration.csv").upload_file(path_df_sp_model_migration, ExtraArgs={'ACL': 'public-read'})
# %%
