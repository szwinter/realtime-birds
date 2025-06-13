# %% Imports
import os
import boto3
import tqdm
import pickle
import numpy as np
import argparse
import rasterio
from rasterio.windows import Window
from matplotlib import pyplot as plt
import pandas as pd

path_project = "/scratch/project_2003104/gtikhono/realtime_birds"
dir_data = "data"
dir_export = "export/spatial_rasters"
df_sp_model = pd.read_csv(os.path.join(path_project, "data", "modeled_species.csv"))
sp_list = list(df_sp_model.species)
factor = 10

j = 8
sp = sp_list[j]
path_sp = os.path.join(path_project, dir_data, "species", sp)


# %%
os.environ["AWS_REQUEST_CHECKSUM_CALCULATION"] = "when_required"
os.environ["AWS_RESPONSE_CHECKSUM_VALIDATION"] = "when_required"
s3_resource = boto3.resource('s3', endpoint_url='https://a3s.fi')
bucket_name = "gtikhono-MKapp-paper-export"
s3_resource.create_bucket(Bucket=bucket_name, ACL="public-read")
my_bucket = s3_resource.Bucket(bucket_name)
for bucket in s3_resource.buckets.all():
    print(bucket.name)

# %% Read priors
path_a_prior = os.path.join(path_sp, sp + "_a.tif")
path_va_prior = os.path.join(path_sp, sp + "_va.tif")
path_b_prior = os.path.join(path_sp, sp + "_b.tif")
with rasterio.open(path_a_prior) as src:
    height_ha, width_ha = src.height, src.width
    transform_ha = src.transform
    height_km = height_ha//factor
    width_km = width_ha//factor
    window = Window(0, 0, width_km*factor, height_km*factor)
    a_prior_ha = src.read(1, window=window)
    height_ha, width_ha = a_prior_ha.shape
    transform_ha = src.window_transform(window) 
    profile = src.profile
    profile.update({"height": height_ha, "width": width_ha,
                    "transform": transform_ha, "nodata": np.nan})

with rasterio.open(path_va_prior) as src:
    va_prior_ha = src.read(1, window=window)
with rasterio.open(path_b_prior) as src:
    b_prior_ha = src.read(1, window=window)
prior_mean_map = np.minimum(a_prior_ha + b_prior_ha, 1)
prior_var_map = va_prior_ha

# %% Read posteriors
path_a_post = os.path.join(path_sp, sp + "_a_app232425.tif")
path_va_post = os.path.join(path_sp, sp + "_va_app232425.tif")
with rasterio.open(path_a_post) as src:
    a_post_ha = src.read(1)
with rasterio.open(path_va_post) as src:
    va_post_ha = src.read(1)
post_mean_map = a_post_ha
post_var_map = va_post_ha

# %%
with np.errstate(divide='ignore'):
    ratio_var_map = 100*(1-post_var_map/prior_var_map)
os.makedirs(os.path.join(path_project, dir_export), exist_ok=True)
path_prior_mean = os.path.join(path_project, dir_export, sp + "_map0_prior_mean.tif")
path_post_mean = os.path.join(path_project, dir_export, sp + "_map1_post_mean.tif")
path_var_reduction = os.path.join(path_project, dir_export, sp + "_map2_var_reduction.tif")

with rasterio.open(path_prior_mean, "w", **profile) as dst:
    dst.write(prior_mean_map, 1)
with rasterio.open(path_post_mean, "w", **profile) as dst:
    dst.write(post_mean_map, 1)
with rasterio.open(path_var_reduction, "w", **profile) as dst:
    dst.write(ratio_var_map, 1)

s3_resource.Object(bucket_name, f'spatial_rasters/{sp + "_map0_prior_mean.tif"}').upload_file(path_prior_mean, ExtraArgs={'ACL': 'public-read'})
s3_resource.Object(bucket_name, f'spatial_rasters/{sp + "_map1_post_mean.tif"}').upload_file(path_post_mean, ExtraArgs={'ACL': 'public-read'})
s3_resource.Object(bucket_name, f'spatial_rasters/{sp + "_map2_var_reduction.tif"}').upload_file(path_var_reduction, ExtraArgs={'ACL': 'public-read'})


# %%

for j, sp in enumerate(tqdm.tqdm(sp_list)):
    path_sp = os.path.join(path_project, dir_data, "species", sp)
    try:
        path_a_prior = os.path.join(path_sp, sp + f"_a.tif")
        path_va_prior = os.path.join(path_sp, sp + f"_va.tif")
        path_b_prior = os.path.join(path_sp, sp + f"_b.tif")
        with rasterio.open(path_a_prior) as src:
            height_ha, width_ha = src.height, src.width
            transform_ha = src.transform
            height_km = height_ha//factor
            width_km = width_ha//factor
            window = Window(0, 0, width_km*factor, height_km*factor)
            a_prior_ha = src.read(1, window=window)
            height_ha, width_ha = a_prior_ha.shape
            transform_ha = src.window_transform(window) 
            profile = src.profile
            profile.update({"height": height_ha, "width": width_ha,
                            "transform": transform_ha, "nodata": np.nan})

        with rasterio.open(path_va_prior) as src:
            va_prior_ha = src.read(1, window=window)
        with rasterio.open(path_b_prior) as src:
            b_prior_ha = src.read(1, window=window)
        prior_mean_map = np.minimum(a_prior_ha + b_prior_ha, 1)
        prior_var_map = va_prior_ha

        path_a_post = os.path.join(path_sp, sp + f"_a_app232425.tif")
        path_va_post = os.path.join(path_sp, sp + f"_va_app232425.tif")
        with rasterio.open(path_a_post) as src:
            a_post_ha = src.read(1)
        with rasterio.open(path_va_post) as src:
            va_post_ha = src.read(1)
        post_mean_map = a_post_ha
        post_var_map = va_post_ha
        
        with np.errstate(divide='ignore'):
            ratio_var_map = 100*(1-post_var_map/prior_var_map)
        os.makedirs(os.path.join(path_project, dir_export), exist_ok=True)
        path_prior_mean = os.path.join(path_project, dir_export, sp + "_map0_prior_mean.tif")
        path_post_mean = os.path.join(path_project, dir_export, sp + "_map1_post_mean.tif")
        path_var_reduction = os.path.join(path_project, dir_export, sp + "_map2_var_reduction.tif")

        with rasterio.open(path_prior_mean, "w", **profile) as dst:
            dst.write(prior_mean_map, 1)
        with rasterio.open(path_post_mean, "w", **profile) as dst:
            dst.write(post_mean_map, 1)
        with rasterio.open(path_var_reduction, "w", **profile) as dst:
            dst.write(ratio_var_map, 1)

        s3_resource.Object(bucket_name, f'spatial_rasters/{sp + "_map0_prior_mean.tif"}').upload_file(path_prior_mean, ExtraArgs={'ACL': 'public-read'})
        s3_resource.Object(bucket_name, f'spatial_rasters/{sp + "_map1_post_mean.tif"}').upload_file(path_post_mean, ExtraArgs={'ACL': 'public-read'})
        s3_resource.Object(bucket_name, f'spatial_rasters/{sp + "_map2_var_reduction.tif"}').upload_file(path_var_reduction, ExtraArgs={'ACL': 'public-read'})

    except:
        print("Failed for %d-%s" % (j,sp))

# %%
