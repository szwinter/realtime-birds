import os
import tqdm
import shutil

path_project = "/scratch/project_2003104/gtikhono/realtime_birds"
dir_orig_data = "orig_data"
dir_data = "data"

for prefix in ["a", "b", "va"]:
  fn = os.listdir(os.path.join(path_project, dir_orig_data,  "%s_maps"%prefix))
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
