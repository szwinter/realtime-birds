import os
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

model_type_list = ["posterior2023", "posterior2024", "posterior2025"]
for model_type in model_type_list:
    reset_prior_detection = reset_prior_migration = reset_prior_spatial = 0
    if model_type == "posterior2023":
        detection_train_range = [1, 365]
        migration_train_range = [1, 365]
        spatial_train_range = [1, 365]
        test_range = [366, 731]
        prior_type = "transect"
        reset_prior_migration = 1
    elif model_type == "posterior2024":
        detection_train_range = [1, 731]
        migration_train_range = [366, 731]
        spatial_train_range = [366, 731]
        test_range = [366, 731]
        prior_type = "app23"
        reset_prior_migration = 0
    elif model_type == "posterior2025":
        detection_train_range = [1, 731]
        migration_train_range = [732, 1096]
        spatial_train_range = [732, 1096]
        test_range = [732, 1096]
        prior_type = "app2324"
        reset_prior_migration = 0
    else:
        raise Exception("Unknown model_type")
    
    suffix_args = [prior_type] + detection_train_range + migration_train_range + spatial_train_range + \
    [reset_prior_detection,reset_prior_migration,reset_prior_spatial] + test_range
    print(suffix_args)
    suffix_result = "%s_det(%d_%d)_mig(%d_%d)_dyn(%d_%d)_test%d%d%d(%d_%d)" % tuple(suffix_args)
    print(suffix_result)
    
    df = pd.DataFrame({"species":sp_list})
    sp = sp_list[0]
    path_result = os.path.join(path_project, dir_results, sp)
    with open(os.path.join(path_result, sp+"_evals_"+suffix_result+".pickle"), "rb") as handle:
        output = pickle.load(handle)
    
    for i, outer_key in enumerate(output):
        if isinstance(output[outer_key], dict):
            for k, inner_key in enumerate(output[outer_key]):
                colname = "%s__%s" % (outer_key, inner_key)
                df[colname] = None
        else:
            df[outer_key] = None
    
    for j, sp in enumerate(tqdm.tqdm(sp_list)):
        path_result = os.path.join(path_project, dir_results, sp)
        try:
            with open(os.path.join(path_result, sp+"_evals_"+suffix_result+".pickle"), "rb") as handle:
                output = pickle.load(handle)
            for i, outer_key in enumerate(output):
                if isinstance(output[outer_key], dict):
                    for k, inner_key in enumerate(output[outer_key]):
                        colname = "%s__%s" % (outer_key, inner_key)
                        df.loc[j, colname] = output[outer_key][inner_key]
                else:
                    df.loc[j, outer_key] = output[outer_key]
        except:
            print("Failed for %d-%s" % (j,sp))
    
    os.makedirs(os.path.join(path_project, "postprocessed"), exist_ok=True)
    df.to_csv(os.path.join(path_project, "postprocessed", "%s.csv" % model_type))
    
    ind = df["prevs__actual"] > 100
    df.columns.values
    plt.scatter(df.loc[ind, "AUCs__prior"], df.loc[ind, "AUCs__GWR_1km"])
    plt.axline([0,0], [1,1], color="red")
    plt.show()
