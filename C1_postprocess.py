import os
import tqdm
import pickle
import numpy as np
import argparse
from matplotlib import pyplot as plt
import pandas as pd


path_project = "/Users/gtikhono/DATA/2025.01.01_realtime_birds/"
dir_results = "results"
sp_list = os.listdir(os.path.join(path_project, dir_results))
sp_list = [name for name in sp_list if os.path.isdir(os.path.join(path_project, dir_results, name))]
sp_list.sort()

parser = argparse.ArgumentParser()
# parser.add_argument("--detstart", type=int, default=1)
# parser.add_argument("--detstop", type=int, default=365)
# parser.add_argument("--migstart", type=int, default=0)
# parser.add_argument("--migstop", type=int, default=0)
# parser.add_argument("--spatstart", type=int, default=1)
# parser.add_argument("--spatstop", type=int, default=365)
# parser.add_argument("--teststart", type=int, default=366)
# parser.add_argument("--teststop", type=int, default=730)
parser.add_argument("--priortype", type=str, default="transect")
parser.add_argument("--savenewprior", type=int, default=0)
parser.add_argument("--savepred", type=int, default=0)
parser.add_argument("--saveimages", type=int, default=0)
parser.add_argument("--factor", type=int, default=10)
parser.add_argument("--jn", type=int, default=4)
args = parser.parse_args()


# detection_train_range = [args.detstart, args.detstop]
# migration_train_range = [args.migstart, args.migstop]
# spatial_train_range = [args.spatstart, args.spatstop]
# test_range = [args.teststart, args.teststop]

detection_train_range = [1, 365]
test_range = [366, 730]
prior_type = args.priortype
save_new_prior = bool(args.savenewprior)
save_prediction = bool(args.savepred)
save_images = bool(args.saveimages)
factor = args.factor
jn = args.jn


model_type_list = ["posterior2023", "posterior2024"]
for model_type in model_type_list:
  if model_type == "posterior2023":
    migration_train_range = [0, 0]
    spatial_train_range = [1, 365]
  elif model_type == "posterior2024":
    migration_train_range = [366, 730]
    spatial_train_range = [366, 730]
  else:
    raise Exception("Unknown model_type")
  
  
  suffix_result = "%s_det(%d_%d)_mig(%d_%d)_dyn(%d_%d)_test(%d_%d)" % tuple([prior_type] + detection_train_range + migration_train_range + spatial_train_range + test_range)
  
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
    # if sp=="alcedo_atthis": aaa
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
