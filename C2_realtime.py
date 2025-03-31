import os
import tqdm
import pickle
import numpy as np
import argparse
from matplotlib import pyplot as plt
import pandas as pd
from utils.eval_utils import fast_auc


path_project = "/scratch/project_2003104/gtikhono/realtime_birds"
dir_results = "results"
sp_list = os.listdir(os.path.join(path_project, dir_results))
sp_list = [name for name in sp_list if os.path.isdir(os.path.join(path_project, dir_results, name))]
sp_list.sort()

parser = argparse.ArgumentParser()
parser.add_argument("--priortype", type=str, default="transect")
parser.add_argument("--factor", type=int, default=10)
parser.add_argument("--trainstep", type=int, default=7)
parser.add_argument("--trainstepnum", type=int, default=52)
parser.add_argument("testwindow", type=int, default=7)
args = parser.parse_args()


prior_type = args.priortype
factor = args.factor
train_step = args.trainstep
step_vec = np.arange(0, args.trainstepnum)
test_window = args.testwindow


detection_train_range = [1, 365]
train_start = 365 + 1
preds_list = [[[] for step in step_vec] for sp in sp_list]
for j, sp in enumerate(tqdm.tqdm(sp_list)):
  for i, step in tqdm.tqdm(step_vec):  
    train_stop = 365 + train_step*step
    migration_train_range = [train_start, train_stop]
    spatial_train_range = [train_start, train_stop]
    test_range = [train_stop+1, train_stop+7]
    suffix_result = "%s_det(%d_%d)_mig(%d_%d)_dyn(%d_%d)_test(%d_%d)" % tuple([prior_type] + detection_train_range + migration_train_range + spatial_train_range + test_range)
    try:
      preds = np.load(os.path.join(path_project, dir_results, sp, sp+"_preds_"+suffix_result+".npy"))
      preds_list[j][i] = preds
    except:
      preds_list[j][i] = np.zeros([0, 6]); #print("Failed for %d-%s" % (j,sp))
  

mat_list = [None] * len(sp_list)
for j, preds in enumerate(tqdm.tqdm(preds_list)):
  preds_all = np.concatenate(preds)
  _, unique_indices_last = np.unique(preds[::-1,0], return_index)
  mat_list[j] = preds_all[unique_indices_last[::-1],:]
# mat_list = [np.concatenate(preds) for preds in tqdm.tqdm(preds_list)]


# df = pd.read_csv(os.path.join(path_project, "postprocessed", "posterior2023.csv"), index_col=0)
# df.iloc[:,1:] = np.nan
df = pd.DataFrame(index=range(len(sp_list), columns=["species", "AUCs__GWR_1km", "AUCs__GWR_1ha", "R2s__GWR_1km", "R2s__GWR_1ha",
                                                     "prevs__GWR_1km", "prevs__GWR_1ha", "llhs__GWR_1km", "llhs__GWR_1ha" ])
df.species = sp_list


def llh(ys, probs):
    probs_clipped = np.clip(probs, 1e-6, 1-1e-6)
    return ys*np.log(probs_clipped) + (1-ys)*np.log(1-probs_clipped)


for j, sp in enumerate(tqdm.tqdm(sp_list)):
  if mat_list[j].shape[0] > 0:
    y = mat_list[j][:,1]
    post_s =  mat_list[j][:,2]
    m_pred =  mat_list[j][:,3]
    post_d_km = mat_list[j][:,4]
    post_d_ha = mat_list[j][:,5]
    post_spatial_km = post_s*m_pred*post_d_km
    post_spatial_ha = post_s*m_pred*post_d_ha
    
    df.loc[j, "AUCs__GWR_1km"] = fast_auc(y, post_spatial_km)
    df.loc[j, "AUCs__GWR_1ha"] = fast_auc(y, post_spatial_ha)
    df.loc[j, "R2s__GWR_1km"] = post_spatial_km[(y==1)].mean() - post_spatial_km[(y==0)].mean()
    df.loc[j, "R2s__GWR_1ha"] = post_spatial_ha[(y==1)].mean() - post_spatial_ha[(y==0)].mean()
    df.loc[j, "prevs__GWR_1km"] = post_spatial_km.sum()
    df.loc[j, "prevs__GWR_1ha"] = post_spatial_ha.sum()
    df.loc[j, "llhs__GWR_1km"] = llh(y, post_spatial_km).mean()
    df.loc[j, "llhs__GWR_1ha"] = llh(y, post_spatial_ha).mean()

df.to_csv(os.path.join(path_project, "postprocessed", "realtime.csv"))


auc_prior_mat = np.nan * np.zeros([len(sp_list), len(step_vec)])
auc_post_mat = np.nan * np.zeros([len(sp_list), len(step_vec)])
llh_prior_mat = np.nan * np.zeros([len(sp_list), len(step_vec)])
llh_post_mat = np.nan * np.zeros([len(sp_list), len(step_vec)])
prev_actual_mat = np.zeros([len(sp_list), len(step_vec)])
prev_prior_mat = np.nan * np.zeros([len(sp_list), len(step_vec)])
prev_post_mat = np.nan * np.zeros([len(sp_list), len(step_vec)])

for j, sp in enumerate(tqdm.tqdm(sp_list)):
  for i, step in enumerate(step_vec):
    # if preds_list[j][i].shape[0] > 0:
    #   y = preds_list[j][i][:,1]
    #   post_s =  preds_list[j][i][:,2]
    #   m_pred =  preds_list[j][i][:,3]
    #   post_d_km = preds_list[j][i][:,4]
    #   post_spatial_km = post_s*m_pred*post_d_km
    #   llh_post_mat[j,i] = llh(y, post_spatial_km).mean()
    #   prev_actual_mat[j,i] = np.sum(preds_list[j][i][:,1])
    #   prev_post_mat[j,i] = np.sum(post_spatial_km)
    train_stop = 365 + train_step*step
    migration_train_range = [train_start, train_stop]
    spatial_train_range = [train_start, train_stop]
    test_range = [train_stop+1, train_stop+test_window]
    suffix_result = "%s_det(%d_%d)_mig(%d_%d)_dyn(%d_%d)_test(%d_%d)" % tuple([prior_type] + detection_train_range + migration_train_range + spatial_train_range + test_range)
    path_result = os.path.join(path_project, dir_results, sp)
    try:
      with open(os.path.join(path_result, sp+"_evals_"+suffix_result+".pickle"), "rb") as handle:
        output = pickle.load(handle)
    except:
      next
    
    auc_prior_mat[j,i] = output["AUCs"]["detection"]
    auc_post_mat[j,i] = output["AUCs"]["GWR_1km"]
    llh_prior_mat[j,i] = output["llhs"]["detection"]
    llh_post_mat[j,i] = output["llhs"]["GWR_1km"]
    prev_actual_mat[j,i] = output["prevs"]["actual"]
    prev_prior_mat[j,i] = output["prevs"]["detection"]
    prev_post_mat[j,i] = output["prevs"]["GWR_1km"]
 
mat_list = [auc_prior_mat, auc_post_mat, llh_prior_mat, llh_post_mat, prev_actual_mat, prev_prior_mat, prev_post_mat]
names_list = ["auc_prior_mat", "auc_post_mat", "llh_prior_mat", "llh_post_mat", "prev_actual_mat", "prev_prior_mat", "prev_post_mat"]
for mat, name in zip(mat_list, names_list):
  df_mat = pd.DataFrame(mat)
  df_mat.columns = step_vec*train_step
  df_mat.insert(0, "species", sp_list)
  df_mat.to_csv(os.path.join(path_project, "postprocessed", "%s.csv"%name))

# j = 0
# plt.plot(step_vec*train_step, llh_post_mat[j]-llh_prior_mat[j])
# plt.axline([0,0], [1,0], color="red", linestyle="--")
# plt.show()
