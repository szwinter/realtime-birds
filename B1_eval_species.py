import os
import tqdm
import pickle
import rasterio
import argparse
import time
from rasterio.windows import Window
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
from scipy.ndimage import gaussian_filter
from scipy.stats import norm as scipy_norm

from utils.model_utils import fit_detection_model, fit_migration_model, m_numpy
from utils.spatial_utils import DistributionMap, DataMap, fit_GWR, binary_search_dec, binary_search_inc
from utils.eval_utils import fast_auc
start_time = time.time()

# %% Paths and constants
# Detection param prior mean and penalty
beta_prec = 1/100.0
beta_mean =  np.array([-2.0, -2.0, 1.0, 0.0, 0.0])
# Migration functional penalty strength
theta_prec = 1/100.0
# maximum neighborhood size and Gaussian kernel radius
r_nh = 5.0
r_kernel = 2.5

path_project = "/scratch/project_2003104/gtikhono/realtime_birds"
dir_orig_data = "orig_data"
dir_data = "data"
dir_results = "results"
df_sp_model = pd.read_csv(os.path.join(path_project, dir_data, "modeled_species.csv"))
sp_list = list(df_sp_model.species)

# %% Handle input parameters
parser = argparse.ArgumentParser()
parser.add_argument('species_index', type=int)
parser.add_argument("--detstart", type=int, default=1)
parser.add_argument("--detstop", type=int, default=365)
parser.add_argument("--migstart", type=int, default=1)
parser.add_argument("--migstop", type=int, default=365)
parser.add_argument("--spatstart", type=int, default=1)
parser.add_argument("--spatstop", type=int, default=365)
parser.add_argument("--teststart", type=int, default=366)
parser.add_argument("--teststop", type=int, default=730)
parser.add_argument("--priortype", type=str, default="")
parser.add_argument("--namenewprior", type=str, default="app")
parser.add_argument("--savenewprior", type=int, default=0)
parser.add_argument("--saveimages", type=int, default=0)
parser.add_argument("--savepred", type=int, default=0)
parser.add_argument("--resetpriordet", type=int, default=0)
parser.add_argument("--resetpriormig", type=int, default=0)
parser.add_argument("--resetpriorspat", type=int, default=0)
parser.add_argument("--factor", type=int, default=10)
parser.add_argument("--jn", type=int, default=4)
args = parser.parse_args()
print(' '.join(f'{k}={v}' for k, v in vars(args).items()))

sp = sp_list[args.species_index]
print(sp)
detection_train_range = [args.detstart, args.detstop]
migration_train_range = [args.migstart, args.migstop]
spatial_train_range = [args.spatstart, args.spatstop]
test_range = [args.teststart, args.teststop]
prior_type = args.priortype
save_new_prior = bool(args.savenewprior)
name_new_prior = args.namenewprior
save_prediction = bool(args.savepred)
save_images = bool(args.saveimages)
reset_prior_detection = bool(args.resetpriordet)
reset_prior_migration = bool(args.resetpriormig)
reset_prior_spatial = bool(args.resetpriorspat)
factor = args.factor
jn = args.jn


path_sp = os.path.join(path_project, dir_data, "species", sp)
path_result = os.path.join(path_project, dir_results, sp)
os.makedirs(path_result, exist_ok=True)
suffix_args = [prior_type] + detection_train_range + migration_train_range + spatial_train_range + \
    [reset_prior_detection,reset_prior_migration,reset_prior_spatial] + test_range
suffix_result = "%s_det(%d_%d)_mig(%d_%d)_dyn(%d_%d)_test%d%d%d(%d_%d)" % tuple(suffix_args)

if prior_type == "transect":
    path_a = os.path.join(path_sp, sp + "_a.tif")
    path_va = os.path.join(path_sp, sp + "_va.tif")
else:
    path_a = os.path.join(path_sp, sp + f"_a_{prior_type}.tif")
    path_va = os.path.join(path_sp, sp + f"_va_{prior_type}.tif")

# %% Loading non-raster data
with open(os.path.join(path_project, dir_data, "XData.pickle"), 'rb') as handle:
    XData = pickle.load(handle)
short = (XData["rec_class"] == "short").to_numpy().astype(int)
long = (XData["rec_class"] == "long").to_numpy().astype(int)
point = ((XData["rec_class"]!="short")&(XData["rec_class"]!="long")).to_numpy().astype(int)
log_duration = XData["log_duration"].to_numpy()
lats = XData["lat"].to_numpy()
lons = XData["lon"].to_numpy()
days = XData["j.date"].to_numpy() # in {1,...,730}, but later migration model expects values in {1,...,365}. Use days%365!
ones = np.ones(XData.shape[0])

with open(os.path.join(path_project, dir_data, "migration_prior_params.pickle"), 'rb') as handle:
    prior_m_params = pickle.load(handle)
prior_m_params_u_days = prior_m_params.loc[sp][-2:].to_numpy()
prior_m_params = prior_m_params.loc[sp][:6].to_numpy()

with open(os.path.join(path_sp, sp+"_prior.pickle"), 'rb') as handle:
    species = pickle.load(handle)
species["prior.s.L"] = pd.Series(scipy_norm.ppf(species["prior.s"]), index=species["prior.s"].index)
if species["prior.s.L"].isna().any() or species["prior.s.L"].isin([np.inf, -np.inf]).any():
    print("NaNs or infs introduced when mapping s -> Phi^{-1}(s)")
y = species["y"].to_numpy()
prior_s = species["prior.s"].to_numpy()
prior_d_a_transect = species["prior.d.a"].to_numpy()
prior_d_b_transect = species["prior.d.b"].to_numpy()
prior_d_transect = np.minimum(prior_d_a_transect + prior_d_b_transect, 1)
prior_m = species["prior.m"].to_numpy()
prior_sL = species["prior.s.L"].to_numpy()
complete = species["complete"].to_numpy()

detection_train_idx = complete*(detection_train_range[0] <= days)*(days <= detection_train_range[1])
migration_train_idx = complete*(migration_train_range[0] <= days)*(days <= migration_train_range[1])
spatial_train_idx = complete*(spatial_train_range[0] <= days)*(days <= spatial_train_range[1])
test_idx = complete*(test_range[0] <= days)*(days <= test_range[1])

# %% Prior predictions
prior_preds = prior_m*prior_s*prior_d_transect
prior_AUC = fast_auc(y[test_idx], prior_preds[test_idx])
prior_R2 = prior_preds[(y==1)*test_idx].mean() - prior_preds[(y==0)*test_idx].mean()
print("Prior AUC:", np.round(prior_AUC,3))
print("Prior R2:", np.round(prior_R2,3))
# with open(os.path.join(path_result, "%s_predict_prior_%s.txt" % (sp, suffix_result)), "w") as f:
#   f.write("AUC %f\nR2 %f" % (prior_AUC, prior_R2))
  
# %% Reading prior spatial predictions
with rasterio.open(path_a) as src:
    height_ha, width_ha = src.height, src.width
    transform_ha = src.transform
    height_km = height_ha//factor
    width_km = width_ha//factor
    window = Window(0, 0, width_km*factor, height_km*factor)
    a_ha = src.read(1, window=window)
    height_ha, width_ha = a_ha.shape
    transform_ha = src.window_transform(window) 
    profile = src.profile
    profile.update({"height": height_ha, "width": width_ha,
                    "transform": transform_ha, "nodata": np.nan})

with rasterio.open(path_va) as src:
	va_ha = src.read(1, window=window)
va_ha[(np.isnan(va_ha))&(~np.isnan(a_ha))] = 1.0 # ensure a_map != nan implies va_map != nan
va_ha[np.isnan(a_ha)] = np.nan
va_ha = np.clip(va_ha, 1e-4, 1) # clip very small variances to avoid numerical issues with precision

# %% Calculating and downscaling grids 
a_km = a_ha.copy()
a_km = a_km.reshape(height_km, factor, width_km, factor)
a_km = np.nanmean(a_km, axis=(1, 3))
va_km = va_ha.copy()
va_km = va_km.reshape(height_km, factor, width_km, factor)
va_km = np.nanmean(va_km, axis=(1, 3))
transform_km = transform_ha*transform_ha.scale(factor, factor)

# extract grids
ys_km = np.arange(height_km)
_, lat_grid_km = rasterio.transform.xy(transform_km, ys_km, np.zeros_like(ys_km))
lat_grid_km = np.array(lat_grid_km)
xs_km = np.arange(width_km)
lon_grid_km, _ = rasterio.transform.xy(transform_km, np.zeros_like(xs_km), xs_km)
lon_grid_km = np.array(lon_grid_km)

ys_ha = np.arange(height_ha)
_, lat_grid_ha = rasterio.transform.xy(transform_ha, ys_ha, np.zeros_like(ys_ha))
lat_grid_ha = np.array(lat_grid_ha)
xs_ha = np.arange(width_ha)
lon_grid_ha, _ = rasterio.transform.xy(transform_ha, np.zeros_like(xs_ha), xs_ha)
lon_grid_ha = np.array(lon_grid_ha)

lat_min, lat_max = max(lat_grid_ha[-1], lat_grid_km[-1]), min(lat_grid_ha[0], lat_grid_km[0]) 
lon_min, lon_max = max(lon_grid_ha[0], lon_grid_km[0]), min(lon_grid_ha[-1], lon_grid_km[-1])
lons_clipped = np.clip(lons, lon_min, lon_max)
lats_clipped = np.clip(lats, lat_min, lat_max)

cell_idx = np.arange(height_km*width_km).reshape(height_km, width_km)
rows_to_grid = rasterio.transform.rowcol(transform_km, lons_clipped, lats_clipped)
rows_to_idx = cell_idx[rows_to_grid]

cell_idx_ha = np.arange(height_ha*width_ha).reshape(height_ha, width_ha)
rows_to_grid_ha = rasterio.transform.rowcol(transform_ha, lons_clipped, lats_clipped)
rows_to_idx_ha = cell_idx_ha[rows_to_grid_ha]

prior_d_a = a_ha.flatten()[rows_to_idx_ha]
prior_d = np.minimum(prior_d_a + prior_d_b_transect, 1) if prior_type == "transect" else prior_d_a

# %% Training detection model and making prediction
tau_detection = prior_m*prior_d
X_detection = np.c_[ones*short, ones*long, prior_sL, short*log_duration, long*log_duration, point]
beta = fit_detection_model(y[detection_train_idx], X_detection[detection_train_idx,:5], 
                           tau_detection[detection_train_idx,], beta_mean, beta_prec)
point_intercept = (beta[0] + beta[1] + (beta[3] + beta[4])*np.log(300))/2 # irrelevant for 2023 but needed for 2024
beta = np.append(beta, point_intercept)
post_s = scipy_norm.cdf(X_detection@beta)

post_detection = post_s*tau_detection
detection_AUC = fast_auc(y[test_idx], post_detection[test_idx])
detection_R2 = post_detection[(y==1)*test_idx].mean() - post_detection[(y==0)*test_idx].mean()
print("AUC after updating detection:", np.round(detection_AUC,3))
print("R2 after updating detection:", np.round(detection_R2,3))
# with open(os.path.join(path_result, "%s_predict_detection_%s.txt" % (sp, suffix_result)), "w") as f:
#   f.write("AUC %f\nR2 %f" % (detection_AUC, detection_R2))

# %% Training migration model and making prediction
if np.var(prior_m) == 0:
    print("Migration model has no variance; skipping resident species.")
    post_m = prior_m
    theta = prior_m_params
elif (migration_train_range[0] == 0 and migration_train_range[1] == 0) or (migration_train_range[1] < migration_train_range[0]):
    print("Specified migration train range forces migration model match prior")
    post_m = prior_m
    theta = prior_m_params
else:
    tau_migration = post_s*prior_d
    theta = fit_migration_model(y[migration_train_idx], lats[migration_train_idx], 
                                days[migration_train_idx]%365, tau_migration[migration_train_idx], 
                                prior_m_params, theta_prec)
    post_m = m_numpy(lats, days%365, theta)
    post_migration = post_m*tau_migration
    migration_AUC = fast_auc(y[test_idx], post_migration[test_idx])
    print("AUC after updating migration:", np.round(migration_AUC,3))
    df = pd.DataFrame(theta[None,:])
    df.columns = ["co.first.1","co.first.2","co.last.1","co.last.2","pm.first","pm.last"]
    df.to_csv(os.path.join(path_result, "%s_migration_%s.csv" % (sp, suffix_result)), index=False)

# %% Precompute indices for spatial models
prior_map = DistributionMap(cell_idx=cell_idx, lat_grid=lat_grid_km, lon_grid=lon_grid_km,
                           mean_map=a_km, var_map=va_km)
post_map = DistributionMap(cell_idx=cell_idx, lat_grid=lat_grid_km, lon_grid=lon_grid_km,
                          mean_map=a_km.copy(), var_map=va_km.copy())

cells_with_data = np.unique(rows_to_idx[spatial_train_idx])
cells_to_update = set()
for c in tqdm.tqdm(cells_with_data, mininterval=10, desc="Calculating neighbouring cells"):
    cells_to_update.update(prior_map.get_nearby_cells(c, r_nh))
cells_to_update = np.array(list(cells_to_update)) 
print("Fraction of cells to update:", np.round(len(cells_to_update)/np.sum(~np.isnan(a_km)),2))

Y_dict = {}
threshold_dict = {}
has_data = np.zeros([prior_map.height, prior_map.width]).astype(bool)
tau_spatial = post_s*post_m
rows_to_idx_obs, y_obs, tau_spatial_obs = rows_to_idx[spatial_train_idx], y[spatial_train_idx], tau_spatial[spatial_train_idx]
cells_with_data, unique_inverse, unique_counts = np.unique(rows_to_idx_obs, return_inverse=True, return_counts=True)
ord0 = np.argsort(cells_with_data)
ord1 = np.argsort(unique_inverse)
sel_list = np.split(ord1, np.cumsum(unique_counts[ord0])[:-1])
has_data[cells_with_data//prior_map.width, cells_with_data%prior_map.width] = True
for c, sel in tqdm.tqdm(zip(cells_with_data, sel_list), mininterval=10, desc="Calculating cell-specific data dictionaries"):
    Y_dict[c], threshold_dict[c] = y_obs[sel], tau_spatial_obs[sel]
  
has_data = has_data.flatten()
data_map = DataMap(Y_dict = Y_dict, threshold_dict = threshold_dict, has_data = has_data)

# %% Fit spatial models
def fit_spatial(cArray, j_ind=0):
    m, v = [np.zeros(len(cArray)) for i in range(2)]
    for i, c0 in enumerate(tqdm.tqdm(cArray, mininterval=10, desc="Fitting spatial model, job %d"%j_ind)):
        row = c0//prior_map.width
        col = c0%prior_map.width
        prior_mean = prior_map.mean_map[row, col]
        prior_prec = 1/prior_map.var_map[row, col] # convert to precision
        neighborhood = prior_map.get_nearby_cells(c0, r_nh)
        cells_to_use = neighborhood[data_map.has_data[neighborhood]]
        if len(cells_to_use) == 0:
            m[i], v[i] = prior_map.mean_map[row, col], prior_map.var_map[row, col]
        else:
            Y_train, n_train, threshold_train = data_map.pool_data(cells_to_use)
            cell_weights = prior_map.calculate_kernel(c0, cells_to_use, r_kernel)    
            weights_train = np.repeat(cell_weights, n_train)
            result = fit_GWR(Y_train, threshold_train, weights_train, prior_mean, prior_prec)
            m[i], v[i] = result["mean"], result["variance"]
    return m, v

cell_to_update_list = [cells_to_update[i::jn] for i in range(jn)]
results = Parallel(n_jobs=jn)(delayed(fit_spatial)(c, j) for j, c in enumerate(cell_to_update_list))
origInd = np.argsort(np.concatenate([np.arange(len(cv))*jn+i for i,cv in enumerate(cell_to_update_list)]))
ma = np.concatenate([res[0] for res in results])[origInd]
va = np.concatenate([res[1] for res in results])[origInd]
if len(cells_to_update) > 0:
    post_map.mean_map[cells_to_update//prior_map.width, cells_to_update%prior_map.width] = ma
    post_map.var_map[cells_to_update//prior_map.width, cells_to_update%prior_map.width] = va

# %% Predict with spatial models
post_d_km = post_map.mean_map.flatten()[rows_to_idx]
post_spatial_km = post_s*post_m*post_d_km
spatial_AUC_km = fast_auc(y[test_idx], post_spatial_km[test_idx])
print("AUC after updating spatial (%.2fkm):"%(0.01*factor**2), np.round(spatial_AUC_km,3))

# %% Upscale results from lower dimensional to original and make predictions
iter_n = 20 # 30 / 2**20 = 0.00003, which shall be sufficeintly small error on the [-inf,inf] scale
prior_ha_mean = a_ha
prior_ha_mean_4d = np.reshape(prior_ha_mean, [prior_map.height,factor,prior_map.width,factor])
post_ha_mean_4d = prior_ha_mean_4d.copy()
post_ha_var_4d = np.reshape(va_ha, [prior_map.height,factor,prior_map.width,factor])
if len(cells_to_update) > 0:
    row_vec = cells_to_update // prior_map.width
    col_vec = cells_to_update % prior_map.width
    avg_vec = post_map.mean_map[row_vec, col_vec]
    prob_grid_stack = prior_ha_mean_4d[row_vec,:,col_vec,:]
    L_stack = scipy_norm.ppf(prob_grid_stack)
    avg_ppf_vec = scipy_norm.ppf(avg_vec)
    delta_min = -15*np.ones([len(cells_to_update)])
    delta_max = 15*np.ones([len(cells_to_update)])
    #delta_min = avg_ppf_vec - np.nanmax(L_stack, (-2,-1)) # this 
    #delta_max = avg_ppf_vec - np.nanmin(L_stack, (-2,-1)) # and this fail due to exact 1 in a_map
    f_min = np.nanmean(scipy_norm.cdf(L_stack + delta_min[:,None,None]), (-2,-1))
    f_max = np.nanmean(scipy_norm.cdf(L_stack + delta_max[:,None,None]), (-2,-1))
    # print(np.nanmax(f_min - avg_vec))
    # print(np.nanmin(f_max - avg_vec))
    for i in tqdm.tqdm(range(iter_n), desc="Downscaling to prior_ha_mean with vectorised binary search"):
        delta_center = (delta_min + delta_max) / 2
        f_center = np.nanmean(scipy_norm.cdf(L_stack + delta_center[:,None,None]), (-2,-1))
        ind_pos = f_center > avg_vec
        ind_neg = np.logical_not(ind_pos)
        delta_min[ind_neg] = delta_center[ind_neg]
        delta_max[ind_pos] = delta_center[ind_pos]

    delta_center = (delta_min + delta_max) / 2
    prob_grid_shifted = scipy_norm.cdf(L_stack + delta_center[:,None,None])
    post_ha_mean_4d[row_vec,:,col_vec,:] = prob_grid_shifted
    post_ha_var_4d[row_vec,:,col_vec,:] = post_map.var_map[row_vec, col_vec, None, None]

post_ha_mean = np.reshape(post_ha_mean_4d, [prior_map.height*factor, prior_map.width*factor])
post_ha_var = np.reshape(post_ha_var_4d, [prior_map.height*factor, prior_map.width*factor])
post_d_ha = post_ha_mean.flatten()[rows_to_idx_ha]
post_spatial_ha = post_s*post_m*post_d_ha
spatial_AUC_ha = fast_auc(y[test_idx], post_spatial_ha[test_idx])
print("AUC after updating spatial (1ha):", np.round(spatial_AUC_ha,3))

# %% Save spatial distribution maps
if save_new_prior:
    with rasterio.open(os.path.join(path_sp, sp + f"_a_{name_new_prior}.tif"), "w", **profile) as dst:
        dst.write(post_ha_mean, 1)
    with rasterio.open(os.path.join(path_sp, sp + f"_va_{name_new_prior}.tif"), "w", **profile) as dst:
        dst.write(post_ha_var, 1)

# %% Save images
if save_images:
    mask = ~np.isnan(prior_map.mean_map)
    smoothed_mask = gaussian_filter(mask.astype(float), sigma=3)  
    bounds = [lon_grid_km.min(), lon_grid_km.max(), lat_grid_km.min(), lat_grid_km.max()]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(prior_map.mean_map, vmin=0, vmax=1, cmap="viridis", extent = bounds)
    ax.contour(smoothed_mask[::-1,:], levels=[0.75], colors="black", linewidths=1.5, extent = bounds) 
    ax.set_aspect(2.4)
    fig.colorbar(im, ax=ax, label="Probability")
    ax.set_title("Prior Mean");
    plt.savefig(os.path.join(path_result, sp+"_prior_"+suffix_result + ".jpeg"))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(post_map.mean_map, vmin=0, vmax=1, cmap="viridis", extent = bounds)
    ax.contour(smoothed_mask[::-1,:], levels=[0.75], colors="black", linewidths=1.5, extent = bounds) 
    ax.set_aspect(2.4)
    fig.colorbar(im, ax=ax, label="Probability")
    ax.set_title("Posterior Mean");
    plt.savefig(os.path.join(path_result, sp+"_post_"+suffix_result + ".jpeg"))
      
    delta_mean = post_map.mean_map - prior_map.mean_map
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(delta_mean, cmap="RdBu_r", vmin=-0.25, vmax=0.25, extent = bounds)
    ax.contour(smoothed_mask[::-1,:], levels=[0.75], colors="black", linewidths=1.5, extent = bounds) 
    ax.set_aspect(2.4)
    cbar = fig.colorbar(im, ax=ax, label="Probability")
    cbar.set_ticks([-0.25, -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25])
    cbar.set_ticklabels([r"$\leq -0.25$", "-0.20", "-0.15", "-0.10", "-0.05", "0", "0.05", "0.10", "0.15", "0.20", r"$\geq 0.25$"])
    ax.set_title("Change in Prior Mean");
    plt.savefig(os.path.join(path_result, sp+"_delta_prior_"+suffix_result + ".jpeg"))
    
    ratio_var = 100*(1-post_map.var_map/prior_map.var_map)
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(ratio_var, cmap="Blues", vmin=0, vmax=100, extent=bounds)
    ax.contour(smoothed_mask[::-1,:], levels=[0.75], colors="black", linewidths=1.5, extent = bounds) 
    ax.set_aspect(2.4)
    cbar = fig.colorbar(im, ax=ax, label="Percent")
    ax.set_title("Percentage Reduction in Variance");
    plt.savefig(os.path.join(path_result, sp+"_delta_var_"+suffix_result + ".jpeg"))
    
    bounds = [24.5, 25.5, 60, 60.5]
    lat_min_hel_idx = binary_search_dec(lat_grid_km, 60)
    lat_max_hel_idx = binary_search_dec(lat_grid_km, 60.5)
    lon_min_hel_idx = binary_search_inc(lon_grid_km, 24.5)
    lon_max_hel_idx = binary_search_inc(lon_grid_km, 25.5)
    
    prior_hel_km = prior_map.mean_map[lat_max_hel_idx:lat_min_hel_idx, lon_min_hel_idx:lon_max_hel_idx]
    mask_hel_km = ~np.isnan(prior_hel_km)
    mask_hel_km = gaussian_filter(mask_hel_km.astype(float), sigma=0.1)  
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(prior_hel_km, vmin=0, vmax=1, cmap="viridis", extent = bounds) 
    ax.contour(mask_hel_km[::-1,:], levels=[0.5], colors="black", linewidths=1.5, extent = bounds) 
    ax.set_aspect(2.4)
    fig.colorbar(im, ax=ax, label="Probability")
    ax.set_title("Prior Mean (1 sqkm)");
    plt.savefig(os.path.join(path_result, sp+"_prior_hel_km_"+suffix_result + ".jpeg"))
    
    post_hel_km = post_map.mean_map[lat_max_hel_idx:lat_min_hel_idx, lon_min_hel_idx:lon_max_hel_idx]
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(post_hel_km, vmin=0, vmax=1, cmap="viridis", extent = bounds) 
    ax.contour(mask_hel_km[::-1,:], levels=[0.5], colors="black", linewidths=1.5, extent = bounds) 
    ax.set_aspect(2.4)
    fig.colorbar(im, ax=ax, label="Probability")
    ax.set_title("Posterior Mean (1 sqkm, not shifted)");
    plt.savefig(os.path.join(path_result, sp+"_post_hel_km_"+suffix_result + ".jpeg"))
    
    bounds = [24.5, 25.5, 60, 60.5]
    lat_min_hel_idx = binary_search_dec(lat_grid_ha, 60)
    lat_max_hel_idx = binary_search_dec(lat_grid_ha, 60.5)
    lon_min_hel_idx = binary_search_inc(lon_grid_ha, 24.5)
    lon_max_hel_idx = binary_search_inc(lon_grid_ha, 25.5)
    
    prior_hel_ha = prior_ha_mean[lat_max_hel_idx:lat_min_hel_idx, lon_min_hel_idx:lon_max_hel_idx]
    mask_hel_ha = ~np.isnan(prior_hel_ha)
    mask_hel_ha = gaussian_filter(mask_hel_ha.astype(float), sigma=1)  
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(prior_hel_ha, vmin=0, vmax=1, cmap="viridis", extent = bounds)
    ax.contour(mask_hel_ha[::-1,:], levels=[0.5], colors="black", linewidths=1.5, extent = bounds) 
    ax.set_aspect(2.6)
    fig.colorbar(im, ax=ax, label="Probability")
    ax.set_title("Prior Mean (1ha)");
    plt.savefig(os.path.join(path_result, sp+"_prior_hel_ha_"+suffix_result + ".jpeg"))
    
    post_hel_ha = post_ha_mean[lat_max_hel_idx:lat_min_hel_idx, lon_min_hel_idx:lon_max_hel_idx]
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(post_hel_ha, vmin=0, vmax=1, cmap="viridis", extent = bounds)
    ax.contour(mask_hel_ha[::-1,:], levels=[0.5], colors="black", linewidths=1.5, extent = bounds) 
    ax.set_aspect(2.6)
    fig.colorbar(im, ax=ax, label="Probability")
    ax.set_title("Posterior Mean (1ha, shifted)");
    plt.savefig(os.path.join(path_result, sp+"_post_hel_ha_"+suffix_result + ".jpeg"))
    
    delta_hel_ha = post_hel_ha - prior_hel_ha
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(delta_hel_ha, cmap="RdBu_r", vmin=-1, vmax=1, extent = bounds)
    ax.contour(mask_hel_ha[::-1,:], levels=[0.5], colors="black", linewidths=1.5, extent = bounds) 
    ax.set_aspect(2.6)
    cbar = fig.colorbar(im, ax=ax, label="Probability")
    ax.set_title("Change in Prior Mean");
    plt.savefig(os.path.join(path_result, sp+"_delta_prior_ha_"+suffix_result + ".jpeg"))
    
    ratio_hel_ha = 100*(1-post_hel_ha/prior_hel_ha)
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(ratio_hel_ha, cmap="Blues", vmin=0, vmax=100, extent=bounds)
    ax.contour(mask_hel_ha[::-1,:], levels=[0.5], colors="black", linewidths=1.5, extent = bounds) 
    ax.set_aspect(2.6)
    cbar = fig.colorbar(im, ax=ax, label="Percent")
    ax.set_title("Percentage Reduction in Variance");
    plt.savefig(os.path.join(path_result, sp+"_delta_var_ha_"+suffix_result + ".jpeg"))

# %% Possibly reset model migration component to prior, e.g. for predictions in different year
if reset_prior_migration:
    print("Setting migration model to prior for predictions")
    pred_m = prior_m
else:
    pred_m = post_m

post_spatial_km = post_s * pred_m * post_d_km
post_spatial_ha = post_s * pred_m * post_d_ha

# %% Compute metrices and likelihoods
def llh(ys, probs):
    probs_clipped = np.clip(probs, 1e-6, 1-1e-6)
    return ys*np.log(probs_clipped) + (1-ys)*np.log(1-probs_clipped)

AUCs = {}
AUCs["prior"] = fast_auc(y[test_idx], prior_preds[test_idx])
AUCs["detection"] = fast_auc(y[test_idx], post_detection[test_idx])
AUCs["GWR_1km"] = fast_auc(y[test_idx], post_spatial_km[test_idx])
AUCs["GWR_1ha"] = fast_auc(y[test_idx], post_spatial_ha[test_idx])

R2s = {}
R2s["prior"] =  prior_preds[(y==1)*test_idx].mean() - prior_preds[(y==0)*test_idx].mean()
R2s["detection"] = post_detection[(y==1)*test_idx].mean() - post_detection[(y==0)*test_idx].mean()
R2s["GWR_1km"] = post_spatial_km[(y==1)*test_idx].mean() - post_spatial_km[(y==0)*test_idx].mean()
R2s["GWR_1ha"] = post_spatial_ha[(y==1)*test_idx].mean() - post_spatial_ha[(y==0)*test_idx].mean()

prevs = {}
prevs["actual"] = y[test_idx].sum()
prevs["detection"] = post_detection[test_idx].sum()
prevs["GWR_1km"] = post_spatial_km[test_idx].sum()
prevs["GWR_1ha"] = post_spatial_ha[test_idx].sum()

llhs = {}
llhs["prior"] = llh(y[test_idx], prior_preds[test_idx]).mean()
llhs["detection"] = llh(y[test_idx], post_detection[test_idx]).mean()
llhs["GWR_1km"] = llh(y[test_idx], post_spatial_km[test_idx]).mean()
llhs["GWR_1ha"] = llh(y[test_idx], post_spatial_ha[test_idx]).mean()

output = {}
output["AUCs"] = AUCs
output["R2s"] = R2s
output["prevs"] = prevs
output["llhs"] = llhs
output["time"] = time.time() - start_time

with open(os.path.join(path_result, sp+"_evals_"+suffix_result+".pickle"), "wb") as handle:
    pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
if save_prediction:
    final_preds = np.c_[np.where(test_idx)[0], y[test_idx], post_s[test_idx], pred_m[test_idx], post_d_km[test_idx], post_d_ha[test_idx]]
    np.save(os.path.join(path_result, sp+"_preds_"+suffix_result+".npy"), final_preds)


print("elapsed %.1f sec" % (time.time() - start_time))
try:
    print("—" * shutil.get_terminal_size()[0])
except:
    print("—" * 40)
