import tqdm
import rasterio
import numpy as np
from matplotlib import pyplot as plt

from scipy.optimize import minimize_scalar

def inverse_haversine(lat_center, lon_center, r, R=6371.0):
    """Get lat/lon of four points r units away from lat/lon center, moving in cardinal directions."""
    lat_rad_center = np.radians(lat_center)
    lon_rad_center = np.radians(lon_center)
    
    lat_rad_min = lat_rad_center - r/R
    lat_rad_max = lat_rad_center + r/R
    lon_rad_min = lon_rad_center + 2*np.arcsin(-np.sin(0.5*r/R)/np.cos(lat_rad_center))
    lon_rad_max = lon_rad_center + 2*np.arcsin(np.sin(0.5*r/R)/np.cos(lat_rad_center))
    
    return np.degrees(lat_rad_min), np.degrees(lat_rad_max), np.degrees(lon_rad_min), np.degrees(lon_rad_max)

def haversine(lat1, lon1, lat2, lon2, R = 6371.0):
    """Geodesic distance on sphere."""
    lon_rad1, lat_rad1 = np.radians(lon1), np.radians(lat1)
    lon_rad2, lat_rad2 = np.radians(lon2), np.radians(lat2)
    hav = np.sin(0.5*(lat_rad2 - lat_rad1 ))**2 + np.cos(lat_rad1)*np.cos(lat_rad2)*np.sin(0.5*(lon_rad2 - lon_rad1))**2
    return 2*R*np.arcsin(np.sqrt(hav))

def binary_search_inc(arr, tgt):
    """Find target in increasing array arr with binary search."""
    lo, hi = 0, len(arr)
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] < tgt:
            lo = mid + 1
        else:
            hi = mid
    return lo

def binary_search_dec(arr, tgt):
    """Find target in decreasing array arr with binary search."""
    lo, hi = 0, len(arr)
    while lo < hi:
        mid = (lo+hi)//2
        if tgt >= arr[mid]:  
            hi = mid
        else: lo = mid+1
    return lo

class DistributionMap():
    def __init__(self, cell_idx, lat_grid, lon_grid, mean_map, var_map, sp_name=None):
        self.cell_idx = cell_idx
        self.lat_grid = lat_grid
        self.lon_grid = lon_grid
        self.mean_map = mean_map
        self.var_map = var_map
        self.sp_name = sp_name
        
        self.height = cell_idx.shape[0]
        self.width = cell_idx.shape[1]
        self.C = self.height*self.width
        
    def cell_to_coord(self, c):
        # get (lat, lon) of the cell index c
        return self.lat_grid[c//self.width], self.lon_grid[c%self.width]
    
    def calculate_kernel(self, c, cells, r_kernel):
        c0_coords = self.cell_to_coord(c)
        lats, lons = self.cell_to_coord(cells)
        cell_dists = haversine(*c0_coords, lats, lons)
        cell_weights = np.exp(-0.5*np.square(cell_dists)/r_kernel**2)
        return cell_weights
    
    def get_nearby_cells(self, c, r):
        # Map cell idx to lat/lon
        lat_center, lon_center = self.cell_to_coord(c)
        
        # get bounding box
        lat_min, lat_max, lon_min, lon_max = inverse_haversine(lat_center, lon_center, r)
        
        # locate lat/long bounds on the regular grid
        # lat_min_idx > lat_max_idx because lat_grid is decreasing
        lat_min_idx = binary_search_dec(self.lat_grid, lat_min)
        lat_max_idx = binary_search_dec(self.lat_grid, lat_max)
        lon_min_idx = binary_search_inc(self.lon_grid, lon_min)
        lon_max_idx = binary_search_inc(self.lon_grid, lon_max)

        # select nearby cells, then disregard those without distribution values (e.g., out of country points)
        neighborhood = self.cell_idx[lat_max_idx:lat_min_idx, lon_min_idx:lon_max_idx]
        mask = ~np.isnan(self.mean_map[lat_max_idx:lat_min_idx, lon_min_idx:lon_max_idx])
        
        return neighborhood[mask]
        
class DataMap():
    def __init__(self, Y_dict, threshold_dict, has_data):
        self.Y_dict = Y_dict
        self.threshold_dict = threshold_dict
        self.has_data = has_data
        
    def pool_data(self, cells):
        Y_train = np.concatenate([self.Y_dict[c] for c in cells])
        n_train = [len(self.Y_dict[c]) for c in cells]
        threshold_train = np.concatenate([self.threshold_dict[c] for c in cells])
        return Y_train, n_train, threshold_train
    
def GWR_loss(da, y, threshold, weights, prior_mean, prior_prec):
    # da -> da + (1-da)*u*b
    # dd = da + (1-da)*u*b
    # probs = dd*threshold, etc
    
    probs = da*threshold
    probs = np.clip(probs, 1e-6, 1-1e-6)
    llh = (y*np.log(probs) + (1 - y)*np.log(1 - probs)).dot(weights)
    penalty = -0.5*prior_prec*(da - prior_mean)**2
    return -(llh + penalty)

def GWR_prec(post_mean, Y_train, threshold_train, weights_train, prior_prec):
    num = Y_train - 2*post_mean*Y_train*threshold_train + np.square(threshold_train)*post_mean**2
    denom = np.square(1-post_mean*threshold_train)*post_mean**2
    post_prec = prior_prec + weights_train.dot(num/denom)
    return post_prec

def fit_GWR(Y_train, threshold_train, weights_train, prior_mean, prior_prec):
    # example model fitting function
    # not finding variance correctly (e.g., second derivative); just returning prior
    optim_output = minimize_scalar(GWR_loss, bounds=(0, 1), method='bounded',
                                   args=(Y_train, threshold_train, weights_train, prior_mean, prior_prec))
    post_mean = optim_output.x
    post_prec = GWR_prec(post_mean, Y_train, threshold_train, weights_train, prior_prec)
    return {"mean":post_mean, "variance":1/post_prec}