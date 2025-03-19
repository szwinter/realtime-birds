import pickle
import os
import pyreadr
import numpy as np

path_project = "/scratch/project_2003104/gtikhono/realtime_birds"
dir_orig_data = "orig_data"
dir_data = "data"

meta = pyreadr.read_r(os.path.join(path_project, dir_orig_data, "meta.RData"))
XData = meta["XData"]
XData["log_duration"] = np.log(XData["duration"]+1e-6)
XData["rec_class"] = ""
XData.loc[XData["rec_type"] == "point", "rec_class"] = "fixed"
XData.loc[(XData["duration"] <= 300)&(XData["rec_type"] != "point"), "rec_class"] = "short"
XData.loc[(XData["duration"] > 300)&(XData["rec_type"] != "point"), "rec_class"] = "long"

with open(os.path.join(path_project, dir_data, "XData.pickle"), "wb") as handle:
    pickle.dump(XData, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
prior_params = meta["migration.pars"]
index_style = dict(zip(prior_params.index, [x.lower().replace(" ", "_") for x in prior_params.index]))
prior_params.rename(index=index_style, inplace=True)
with open(os.path.join(path_project, dir_data, "migration_prior_params.pickle"), "wb") as handle:
    pickle.dump(prior_params, handle, protocol=pickle.HIGHEST_PROTOCOL)