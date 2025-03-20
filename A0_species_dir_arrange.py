import os
import tqdm
import shutil
import pyreadr
import pickle
import pandas as pd
import cmd

path_project = "/scratch/project_2003104/gtikhono/realtime_birds"
dir_orig_data = "orig_data"
dir_data = "data/species"
displaywidth = 100

path_priors = os.path.join(path_project, dir_orig_data, "prior predictions")
fn = os.listdir(path_priors)
fn.sort()
cli = cmd.Cmd()
cli.columnize(fn, displaywidth=displaywidth)
for f in tqdm.tqdm(fn):
  x = f.rfind("_")
  sp = f[:x]
  sp_new = sp.lower().replace(" ", "_")
  species_raw = pyreadr.read_r(os.path.join(path_priors, sp+"_prior.RData"))
  species = pd.concat([species_raw[k] for k in species_raw.keys()], axis=1)
  species["complete"] = species.isna().sum(axis=1) == 0
  os.makedirs(os.path.join(path_project, dir_data, sp_new), exist_ok=True)
  with open(os.path.join(path_project, dir_data, sp_new, sp_new+"_prior.pickle"), "wb") as handle:
    pickle.dump(species, handle, protocol=pickle.HIGHEST_PROTOCOL)

for prefix in ["a", "b", "va"]:
  fn = os.listdir(os.path.join(path_project, dir_orig_data,  "%s_maps"%prefix))
  fn.sort()
  cli.columnize(fn, displaywidth=displaywidth)
  for f in tqdm.tqdm(fn):
    x = f.rfind("_")
    sp = f[:x]
    sp_new = sp.lower().replace(" ", "_")
    os.makedirs(os.path.join(path_project, dir_data, sp_new), exist_ok=True)
    file_orig = os.path.join(path_project, dir_orig_data, "%s_maps"%prefix, "%s_%s.tif"%(sp, prefix))
    if prefix == "va":
      suffix_new = prefix+"L"
    else:
      suffix_new = prefix
      file_new = os.path.join(path_project, dir_data, sp_new, "%s_%s.tif"%(sp_new, suffix_new))
      shutil.copyfile(file_orig, file_new)
