import os
import tqdm
import rasterio
import numpy as np
from matplotlib import pyplot as plt
from torch.distributions import Normal
import torch
import argparse

path_project = "/scratch/project_2003104/gtikhono/realtime_birds"
dir_data = "data/species"
path_data = os.path.join(path_project, dir_data)

parser = argparse.ArgumentParser()
parser.add_argument('species_id', type=int)
parser.add_argument("-n", type=int, default=100)
args = parser.parse_args()
sp_id = args.species_id
n_mc = args.n

sp_list = os.listdir(path_data)
sp_list.sort()
sp_dir = sp_list[args.species_id]
print("Calculating vaL for id %d, species %s" % (sp_id, sp_dir))

with rasterio.open(os.path.join(path_data, sp_dir, sp_dir+"_a.tif")) as src:
    a_map, profile = src.read(1), src.profile

with rasterio.open(os.path.join(path_data, sp_dir, sp_dir+"_vaL.tif")) as src:
    vaL_map = src.read(1)

dn = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
vaL_map[(np.isnan(vaL_map))&(~np.isnan(a_map))] = 1.0 # ensure a_map != nan implies va_map != nan
vaL_map[np.isnan(a_map)] = np.nan
aL_map = dn.icdf(torch.tensor(a_map)).numpy()
idx = ~np.isnan(vaL_map)

va_map = np.nan * vaL_map
aL1, vaL1_sqrt = torch.tensor(aL_map[idx]), torch.tensor(np.sqrt(vaL_map[idx]))
dn = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
E_Phi, E_Phi_squared = torch.zeros_like(aL1), torch.zeros_like(vaL1_sqrt)
for _ in tqdm.tqdm(range(n_mc)):
    L_sample = aL1 + vaL1_sqrt*dn.sample()
    p_sample = dn.cdf(torch.tensor(L_sample))
    E_Phi += p_sample
    E_Phi_squared += p_sample**2

E_Phi /= n_mc
E_Phi_squared /= n_mc
va_map[idx] = (E_Phi_squared - E_Phi**2).numpy()

with rasterio.open(os.path.join(path_data, sp_dir, sp_dir+"_va.tif"), "w", **profile) as dst:
    dst.write(va_map.astype(np.float32), 1)
