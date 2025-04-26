import os
import tqdm
import shutil
import pyreadr
import pickle
import pandas as pd
import cmd
import numpy as np

path_project = "/scratch/project_2003104/gtikhono/realtime_birds"
dir_orig_data = "orig_data"
dir_data = "data/species"
displaywidth = 100

path_priors = os.path.join(path_project, dir_orig_data, "prior predictions")
fn = os.listdir(path_priors)
fn.sort()
cli = cmd.Cmd()
cli.columnize(fn, displaywidth=displaywidth)
print("reading meta.RData", flush=True)
meta = pyreadr.read_r(os.path.join(path_project, dir_orig_data, "meta.RData"))
XData = meta["XData"]
print("computing d23, d24, d25", flush=True)
d23 = (XData["j.date"] >= 1+0*365) & (XData["j.date"] <= 1*365)
d24 = (XData["j.date"] >= 1+1*365) & (XData["j.date"] <= 2*365)
d25 = (XData["j.date"] >= 1+2*365) & (XData["j.date"] <= 3*365)
df_sp = pd.DataFrame(index=np.arange(len(fn)), columns=["species", "p23", "p24", "p25"])
for j, f in enumerate(tqdm.tqdm(fn)):
    x = f.rfind("_")
    sp = f[:x]
    sp_new = "%.3d_%s" % (j, sp.lower().replace(" ", "_"))
    df_sp.loc[j, "species"] = sp_new
    species_raw = pyreadr.read_r(os.path.join(path_priors, sp+"_prior.RData"))
    species = pd.concat([species_raw[k] for k in species_raw.keys()], axis=1)
    species["complete"] = species.isna().sum(axis=1) == 0
    species["prior.s"] = np.clip(species["prior.s"], 1e-6, 1-1e-6)
    df_sp.loc[j, "p23"] = ((species.y==1) & d23).sum()
    df_sp.loc[j, "p24"] = ((species.y==1) & d24).sum()
    df_sp.loc[j, "p25"] = ((species.y==1) & d25).sum()
    os.makedirs(os.path.join(path_project, dir_data, sp_new), exist_ok=True)
    with open(os.path.join(path_project, dir_data, sp_new, sp_new+"_prior.pickle"), "wb") as handle:
        pickle.dump(species, handle, protocol=pickle.HIGHEST_PROTOCOL)
df_sp.to_csv(os.path.join(path_project, "data", "presence_counts.csv"), index=False)

for prefix in ["a", "b", "va"]:
    # fn = os.listdir(os.path.join(path_project, dir_orig_data,  "%s_maps"%prefix))
    fn = os.listdir(path_priors)
    fn.sort()
    # cli.columnize(fn, displaywidth=displaywidth)
    for j, f in enumerate(tqdm.tqdm(fn)):
        x = f.rfind("_")
        sp = f[:x]
        sp_new = "%.3d_%s" % (j, sp.lower().replace(" ", "_"))
        file_orig = os.path.join(path_project, dir_orig_data, "%s_maps"%prefix, "%s_%s.tif"%(sp, prefix))
        suffix_new = prefix+"L" if prefix == "va" else prefix
        file_new = os.path.join(path_project, dir_data, sp_new, "%s_%s.tif"%(sp_new, suffix_new))
        shutil.copyfile(file_orig, file_new)
