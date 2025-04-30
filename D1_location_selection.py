import tqdm
import pickle
import sys
import os
import pyreadr
import argparse
import numpy as np
import pandas as pd
from pyproj import Geod
from itertools import chain
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Patch
import rasterio
from rasterio.transform import Affine
from rasterio.windows import from_bounds
from datetime import datetime
from utils.model_utils import m_numpy
from utils.eval_utils import fast_auc
np.random.seed(123)

# -------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--spnumber', type=int, default=20)
parser.add_argument('--area', type=str, default="all")
parser.add_argument("--priortype", type=str, default="app20232024")
args = parser.parse_args()
print(' '.join(f'{k}={v}' for k, v in vars(args).items()), flush=True)
spN = args.spnumber
area = args.area
priortype = args.priortype
path_project = "/scratch/project_2003104/gtikhono/realtime_birds"
dir_orig_data = "orig_data"
dir_data = "data"
df_sp_model = pd.read_csv(os.path.join(path_project, dir_data, "modeled_species.csv"))
spnames = list(df_sp_model.sort_values("p24", ascending=False).species[:min(df_sp_model.shape[0], spN)])
p = len(spnames)
print(p, spnames)

# -------------------------------------------------------------------------------------------------
sci_data = pd.read_csv("/users/gtikhono/realtime-birds/data/baselocations.csv")
if area == "all":
    min_lon, max_lon = np.min(sci_data.lon)-0.5, np.max(sci_data.lon)+1.1
    min_lat, max_lat = np.min(sci_data.lat)-0.5, np.max(sci_data.lat)+0.5
elif area == "south":
    min_lon, max_lon = 20.9, 27
    min_lat, max_lat = 59.8, 62.8
elif area == "north":
    min_lon, max_lon = 24.3, 29.5
    min_lat, max_lat = 64, 65.4
else:
    raise("Incorrect area parameter - can be all, south and north")

ind = (sci_data.lon >= min_lon) & (sci_data.lon <= max_lon) & (sci_data.lat >= min_lat) & (sci_data.lat <= max_lat)
sci_data = sci_data.loc[ind].reset_index()
print(sci_data)
n_sci = sci_data.shape[0]
T = sci_data.n.max() * 5 + 3
sci_days = np.zeros([n_sci, T], dtype=int)
for i in range(n_sci):
    ind = np.arange(5*sci_data.n[i]+3)
    ind = np.round(np.linspace(0, T-1, 5*sci_data.n[i]+3)).astype(int)
    sci_days[i, ind] = 1 

r_search = 50
sites_per_day = 10
r_day = 10
r_min = 0.5 # change to 0.5

tau = 400 # sigma = 1/sqrt(tau)
gamma = 0.95
res = 0.01

# -------------------------------------------------------------------------------------------------
path_migration = os.path.join(path_project, dir_data, "migration_prior_params.pickle")
with open(path_migration, "rb") as handle:
    m_params = pickle.load(handle)
index_style = dict(zip(m_params.index, [x.lower().replace(" ", "_") for x in m_params.index]))
m_params.rename(index=index_style, inplace=True)

# -------------------------------------------------------------------------------------------------
sp = spnames[0]
path_sp = os.path.join(path_project, dir_data, "species", sp)
with rasterio.open(os.path.join(path_sp, sp + "_a.tif")) as src:
    window = from_bounds(min_lon, min_lat, max_lon, max_lat, transform=src.transform)
    height, width = window.round_lengths().height, window.round_lengths().width
    transform = src.window_transform(window)

a_prior = np.zeros([height, width, p])
a_post = np.zeros([height, width, p])
for j in tqdm.tqdm(range(p)):
    sp = spnames[j]
    path_sp = os.path.join(path_project, dir_data, "species", sp)
    with rasterio.open(os.path.join(path_sp, sp + "_a.tif")) as src:
        a_prior[:,:,j] = src.read(1, window=window)      
    with rasterio.open(os.path.join(path_sp, sp + f"_a_{priortype}.tif")) as src:
        a_post[:,:,j] = src.read(1, window=window)

delta = np.abs(a_prior - a_post)
_, lat_grid = rasterio.transform.xy(transform, np.arange(height), np.zeros(height))
lon_grid, _ = rasterio.transform.xy(transform, np.zeros(width), np.arange(width))

# -------------------------------------------------------------------------------------------------
coords_sci = []
idx_sci = []
in_bounds = ~np.isnan(a_prior).any(axis=2)
for i in range(n_sci):
    lon, lat = sci_data.loc[i, ["lon","lat"]]
    row, col = rasterio.transform.rowcol(transform, lon, lat)
    coords_sci.append([lon, lat])
    idx_sci.append([row, col])
coords_sci = np.array(coords_sci)
idx_sci = np.array(idx_sci)

# -------------------------------------------------------------------------------------------------
def make_mask(R, scale=10):
    R_cells = int(scale * R)
    y, x = np.ogrid[-R_cells:R_cells+1, -R_cells:R_cells+1]
    mask = (x**2 + y**2) <= R_cells**2
    return mask

def apply_mask(target, row, col, mask, invert=False, inplace=False):
    if inplace == False:
        out = target.copy()
    else:
        out = target
    H, W = target.shape
    k = mask.shape[0]
    R = k // 2
    r0 = max(0, row - R)
    r1 = min(H, row + R + 1)
    c0 = max(0, col - R)
    c1 = min(W, col + R + 1)
    mr0 = R - (row - r0)
    mr1 = mr0 + (r1 - r0)
    mc0 = R - (col - c0)
    mc1 = mc0 + (c1 - c0)
    m = mask[mr0:mr1, mc0:mc1]
    if invert:
        out[r0:r1, c0:c1] &= ~m
    else:
        out[:] = 0
        out[r0:r1, c0:c1] = target[r0:r1, c0:c1] & m
    if inplace == False:
        return out    

search_circle = make_mask(r_search)
sci_mask = np.ones([height, width, n_sci], bool)
for i in tqdm.tqdm(range(n_sci)):
    sci_mask[:,:,i] = apply_mask(sci_mask[:,:,i], idx_sci[i,0], idx_sci[i,1], search_circle)
sci_mask = sci_mask * in_bounds[:,:,None]
search_map = sci_mask.any(axis=2)
day_circle = make_mask(r_day)
site_circle = make_mask(r_min)

# -------------------------------------------------------------------------------------------------
cmap_base = ListedColormap(['#BEF2FF', '#4CB648'])
circle_cmap = ListedColormap(['#858383'])
fig, ax = plt.subplots()
ax.imshow(in_bounds, cmap=cmap_base, origin='upper')
alpha_mask = np.ma.masked_where(~search_map, search_map)
ax.imshow(alpha_mask, cmap=circle_cmap, alpha=1, origin='upper')
y, x = rasterio.transform.rowcol(transform, coords_sci[:,0], coords_sci[:,1])
ax.plot(x, y, 'o', color='red', markersize=5)
plt.axis('off')
plt.tight_layout()
plt.show()

# -------------------------------------------------------------------------------------------------
sci_id, idx = [None]*T, [None]*T
global_mask = in_bounds.copy() # prevents searching near previously searched sites

grid_vals = np.arange(0, 1 + res, res) 
grid_max_idx = len(grid_vals) - 1
log_kernel_grid = np.zeros((len(grid_vals), len(grid_vals), p))
grid_idx_prior = np.clip(np.round(a_prior / res).astype(int), 0, grid_max_idx)
grid_idx_post = np.clip(np.round(a_post / res).astype(int), 0, grid_max_idx)
jday_0 = 121
migration_threshold = 0.75

repulsion_grids = []
repulsion = np.ones_like(a_prior, dtype=np.float32)
for t in range(T):
    m_mat = np.stack([m_numpy(lat_grid, jday_0+t, m_params.loc[spnames[j]][:6].to_numpy()) for j in range(p)], axis=1)
    m_mask = m_mat > migration_threshold
    m_mask = np.broadcast_to(m_mask[:,None,:], delta.shape)
    U_active = np.sum(delta * m_mask * repulsion, axis=2)
    
    sci_id_day, idx_day = [None] * n_sci, [None] * n_sci
    # should permute this order across days, np.shuffle
    ind_sci_active = np.where(sci_days[:,t])[0]
    for i in tqdm.tqdm(ind_sci_active, desc=f"Day {t+1}/{T}"):
        sci_id_day[i] = i #sci_data.person_code[i]
        idx_i = np.zeros([sites_per_day, 2], int)
        # Get the first study location for scientist i
        U_local = U_active * sci_mask[:,:,i] * global_mask
        idx_i[0] = np.unravel_index(np.nanargmax(U_local), [height,width])
        global_mask = apply_mask(global_mask, idx_i[0,0], idx_i[0,1], site_circle, invert=True)
        # exclude cells further than r_day away
        local_mask = apply_mask(sci_mask[:,:,i], idx_i[0,0], idx_i[0,1], day_circle)
        U_active_local = U_active * local_mask
        # search for additional cells in a neighborhood
        for k in range(1, sites_per_day):
            U_local = U_active_local * global_mask
            idx_i[k] = np.unravel_index(np.nanargmax(U_local), [height,width])
            global_mask = apply_mask(global_mask, idx_i[k,0], idx_i[k,1], site_circle, invert=True)
        idx_day[i] = idx_i

    sci_id[t] = [sci_id_day[i] for i in ind_sci_active]
    idx[t] = [idx_day[i] for i in ind_sci_active]

    idx_arr = np.concatenate(idx[t])
    prior_last = a_prior[idx_arr[:,0], idx_arr[:,1], :]
    post_last = a_post[idx_arr[:,0], idx_arr[:,1], :]
    m_mask_last = m_mask[idx_arr[:,0], idx_arr[:,1], :]
    d2_prior = np.square(grid_vals[:,None,None] - prior_last[None,:,:])
    d2_post  = np.square(grid_vals[:,None,None] - post_last[None,:,:])
    d2 = d2_prior[:,None,:,:] + d2_post[None,:,:,:] # res**-2 x M x P
    log_kernel = np.log1p(-gamma * np.exp(-0.5 * tau * d2)) * m_mask_last
    log_kernel_grid = 1*log_kernel_grid + np.sum(log_kernel, axis=2)

    kernel_grid = np.exp(log_kernel_grid)
    repulsion = np.empty_like(a_prior, dtype=np.float32)
    for j in range(p):
        repulsion[:,:,j] = kernel_grid[grid_idx_prior[:,:,j], grid_idx_post[:,:,j], j]
    repulsion_grids.append(kernel_grid)
    
# -------------------------------------------------------------------------------------------------
def collect_results(idx, sci_id):
    records = []
    T = len(idx)
    for t in range(T):
        for i in range(len(idx[t])):
            sid = sci_id[t][i]
            for j, rc in enumerate(idx[t][i]):
                row, col = rc
                lat, lon = lat_grid[row], lon_grid[col]
                records.append((t, sid, j, lat, lon, row, col))
    return pd.DataFrame(records, columns=["t", "sci_id", "exp_id", "lat", "lon", "row", "col"])

res = collect_results(idx, sci_id)
res.insert(np.where(res.columns=="sci_id")[0][0], "person_code",  sci_data.loc[res.sci_id, "person_code"].reset_index(drop=True))
migration_sel = pd.DataFrame(dict(zip(spnames, [m_numpy(res.lat.values, jday_0+res.t, m_params.loc[spnames[j]][:6].to_numpy()) for j in range(p)])))
prior_sel = pd.DataFrame(a_prior[res["row"], res["col"], :], columns=spnames)
post_sel = pd.DataFrame(a_post[res["row"], res["col"], :], columns=spnames)
timestamp = datetime.now().strftime('%m%d_%H%M%S')
res_filename = f"/users/gtikhono/realtime-birds/data/sel_loc_{area}_{priortype}_sp{p:03d}_{timestamp}.csv"
migration_filename = f"/users/gtikhono/realtime-birds/data/migration_{area}_{priortype}_sp{p:03d}_{timestamp}.csv"
prior_filename = f"/users/gtikhono/realtime-birds/data/prior_{area}_{priortype}_sp{p:03d}_{timestamp}.csv"
post_filename = f"/users/gtikhono/realtime-birds/data/post_{area}_{priortype}_sp{p:03d}_{timestamp}.csv"
res.to_csv(res_filename, index=False)
migration_sel.to_csv(migration_filename, index=False)
prior_sel.to_csv(prior_filename, index=False)
post_sel.to_csv(post_filename, index=False)
print("saved", res_filename)

# -------------------------------------------------------------------------------------------------
geod = Geod(ellps="WGS84")
az = np.r_[0, 45 + np.arange(4)*90]
dist = np.r_[0, np.sqrt(2)*50*np.ones([4])]
df_list = [None] * res.shape[0]
res.drop(["row","col"], axis=1, inplace=True)
for i in tqdm.tqdm(range(res.shape[0])):
    df = pd.concat([res.loc[i:i]]*5).reset_index(drop=True)
    df.loc[:, ["lon","lat"]] = np.transpose(np.array(geod.fwd(df.lon, df.lat, az, dist)[:2]))
    df["point"] = np.arange(df.shape[0])
    df["id"] = [f"{df.person_code[k]}_d{df.t[k]:02}_s{df.exp_id[k]}_p{df.point[k]}" for k in range(df.shape[0])]
    df_list[i] = df

res_ext = pd.concat(df_list)
ext_filename = f"/users/gtikhono/realtime-birds/data/sel_loc_ext_{area}_{priortype}_sp{p:03d}_{timestamp}.csv"
res_ext.to_csv(ext_filename, index=False)

# -------------------------------------------------------------------------------------------------
valid_indices = np.argwhere(U_local > 0)
row, col = valid_indices[np.random.choice(len(valid_indices))]
res["row"] = row
res["col"] = col
res["lat"] = lat_grid[row]
res["lon"] = lon_grid[col]
prior_sel = pd.DataFrame(a_prior[res["row"], res["col"], :], columns=spnames)
post_sel = pd.DataFrame(a_post[res["row"], res["col"], :], columns=spnames)
res_filename = f"/users/gtikhono/realtime-birds/data/rand_sel_loc_{area}_{priortype}_sp{p:03d}_{timestamp}.csv"
prior_filename = f"/users/gtikhono/realtime-birds/data/rand_prior_{area}_{priortype}_sp{p:03d}_{timestamp}.csv"
post_filename = f"/users/gtikhono/realtime-birds/data/rand_post_{area}_{priortype}_sp{p:03d}_{timestamp}.csv"
res.to_csv(res_filename, index=False)
prior_sel.to_csv(prior_filename, index=False)
post_sel.to_csv(post_filename, index=False)

# -------------------------------------------------------------------------------------------------
