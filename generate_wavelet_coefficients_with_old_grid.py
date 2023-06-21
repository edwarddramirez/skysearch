import timeit as timeit
import sys
import numpy as np
import _wavelets as wt
import os

ti = timeit.default_timer()

# load local directory
username="ramirez"
local_dir = "/het/p4/"+username+"/gcewavelets/skysearch/"
maps_dir = "/het/p4/"+username+"/gcewavelets/skysearch/data/maps/"

# ---
 # Block: Specify map ids 
map_dir = maps_dir + 'map_test/'
data_dir = map_dir + 'projected_maps/' 
# ---

# load father pixel and data
npix = int( float(sys.argv[1]) )
patch_dir = data_dir + 'map_' + str(npix) + '/'
grid_scale_deg = float(sys.argv[4])
str_grid_scale_deg = str.format('{0:.4f}',grid_scale_deg)
projected_map_name = 'projected_map_dict_with_grid_' + str_grid_scale_deg + '_' + '.npz'
projected_map = dict(np.load(patch_dir + projected_map_name))

# --------------------------------------------------------------------------------------
# Ideally, you should reshape the data in the data processing step, have all the points ready, and compute the coefficients
# This would be reduced to a single line. Not worth pursuing right now though (see 04-21-2023 notes)
# data_search = projected_map['search_region_points']
# data_outer = projected_map['outer_region_points']
# data_outmost = projected_map['outmost_region_points']

# data_search = np.swapaxes(data_search, -1, -2).copy() (modified proj_ fcts and outputs in _maps.py and generate_patches.py)
# data_outer = np.swapaxes(data_outer, -1, -2).copy()
# data_outmost = np.swapaxes(data_outmost, -1, -2).copy()

# data_search_and_outer = np.concatenate((data_search, data_outer), axis = 0) (this step now occurs in generate_patches.py code)
# data_redundant = np.concatenate((data_search_and_outer, data_outmost), axis = 0)

# _, unique_indices = np.unique(data_redundant, axis = 0, return_index = True)
# data = data_redundant[unique_indices,:]
data = projected_map['all_points']
# --------------------------------------------------------------------------------------
mesh_bxby = projected_map['rectangular_grid']
grid = projected_map['grid']

# load scale parameter
a_deg = float(sys.argv[2])
a = a_deg * np.pi / 180
arr_a = np.array([a])
Na = len(arr_a)

# define 2d wavelet
wavelet_name = sys.argv[3]
mexh = wt._2d_wavelet(wavelet_name)

# define range of translations (b) and scales (a) to compute cwt
results_dir = patch_dir

N_data, N_dim = data.shape
Ny, Nx = mesh_bxby.shape[:-1]
coefficient_map = np.zeros((Ny, Nx, 1))

buf_data = data[np.newaxis,np.newaxis,np.newaxis]
buf_mesh_bxby = mesh_bxby[grid][:,np.newaxis,np.newaxis,np.newaxis]
buf_arr_a = arr_a[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]

# buf_grid = grid[:,:,np.newaxis,np.newaxis,np.newaxis]
# print(buf_grid.shape)
# print(buf_mesh_bxby[buf_grid].shape)
buf_arr_arg_grid =  ( (buf_data - buf_mesh_bxby ) / buf_arr_a )
# print(buf_arr_arg_grid.shape)

buf_mexh_output_grid = mexh.base_fct(buf_arr_arg_grid)
# print(buf_mexh_output_grid.shape)
# remove two dimensions of a-array to divide mexh
buf_arr_a_sq = np.squeeze(buf_arr_a, axis = -1)
buf_arr_a_sq = np.squeeze(buf_arr_a_sq, axis = -1)

# estimate wavelet coefficient through sum
coefficient_map_flat = np.sum(buf_mexh_output_grid, axis = -1) / buf_arr_a_sq / N_data
coefficient_map_flat_sq = np.squeeze(coefficient_map_flat, axis = -1)
coefficient_map[grid] = coefficient_map_flat_sq
# print(coefficient_map.shape)

# # save coefficient estimate
# a_min = np.min(arr_a) ; a_max = np.max(arr_a) ; Na = len(Na)
# str_a_min = str.format('{0:.3f}',a_min) ; str_a_max = str.format('{0:.3f}',a_max) ; str_Na = str(Na)
# file_name = 'coefficient_map' + '_' + str_a_min + '_' + str_a_max + '_' + str_Na
str_a_deg = str.format('{0:.5f}',a_deg)
file_name = wavelet_name + '_' + 'coefficient_map' + '_' + str_a_deg + '_' + str_grid_scale_deg

np.save(patch_dir + file_name, coefficient_map)

tf = timeit.default_timer()
print('Run Time (min): ' + str( (tf - ti) / 60 ) )