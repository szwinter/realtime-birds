# %% Imports
import pickle
import os
import pyreadr
import numpy as np
import pandas as pd


path_project = "/scratch/project_2003104/gtikhono/realtime_birds"
dir_orig_data = "orig_data"
dir_data = "data"

# %% read meta.RData
meta = pyreadr.read_r(os.path.join(path_project, dir_orig_data, "meta.RData"))
XData = meta["XData"]

# %% prepare XData
XData["log_duration"] = np.log10(XData["duration"])
XData["rec_class"] = ""
XData.loc[XData["rec_type"] == "point", "rec_class"] = "fixed"
XData.loc[(XData["duration"] <= 300)&(XData["rec_type"] != "point"), "rec_class"] = "short"
XData.loc[(XData["duration"] > 300)&(XData["rec_type"] != "point"), "rec_class"] = "long"

os.makedirs(os.path.join(path_project, dir_data), exist_ok=True)
with open(os.path.join(path_project, dir_data, "XData.pickle"), "wb") as handle:
    pickle.dump(XData, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
# %% prepare migration parameters
df_sp = pd.read_csv(os.path.join(path_project, "data", "presence_counts.csv"))
prior_params = meta["migration.pars"]
prior_params.sort_index(inplace=True)
index_style = dict(zip(prior_params.index, df_sp.species))
prior_params.rename(index=index_style, inplace=True)
with open(os.path.join(path_project, dir_data, "migration_prior_params.pickle"), "wb") as handle:
    pickle.dump(prior_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

# %% prepare list of modeled species
threshold_presence = 20
df_sp_model = df_sp.loc[df_sp.p24 >= threshold_presence]
df_sp_model.reset_index(drop=True, inplace=True)
display(df_sp_model)
df_sp_model.to_csv(os.path.join(path_project, "data", "modeled_species.csv"), index=False)