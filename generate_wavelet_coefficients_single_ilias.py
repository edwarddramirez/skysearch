import timeit as timeit
import sys
import numpy as np
import _wavelets as wt
import os
from tqdm import tqdm

ti = timeit.default_timer()

# load local directory
username="ramirez"
local_dir = "/het/p4/"+username+"/gcewavelets/skysearch/"
maps_dir = "/het/p4/"+username+"/gcewavelets/skysearch/data/maps/"

# load map type (with ids corresponding to energy bin, random iteration, bkgd/ps ids)
model = sys.argv[1] # only available option: SA0
trial_id = sys.argv[2]
model_dir = maps_dir + (model + '_' + trial_id + '/')
energy_bin = sys.argv[3]
map_type = sys.argv[4]

# inj_ids
inj_id_list = [1008,1009,1010,1015,1020,1025,1030,1035]
N_counts_list = [8,9,10,15,20,25,30,35]

for m in tqdm(range(len(inj_id_list))):
    inj_id = str(inj_id_list[m])
    
    # load events from map 
    if energy_bin == 'all' or energy_bin == str(-1):
        if map_type == 'bkgd':
            map_dir = model_dir + 'bkgd/'
        elif map_type == 'all' or map_type == 'ps':
            map_dir = model_dir + ('bkgd_wps_' + inj_id + '/') 
    else:
        ie = int(float(energy_bin))
        if map_type == 'bkgd':
            bkgd_dir = model_dir + 'bkgd/'
            map_dir = bkgd_dir + 'energy_bin_' + str(ie) + '/'
        elif map_type == 'all' or map_type == 'ps':
            bkgd_wps_dir = model_dir + ('bkgd_wps_' + inj_id + '/')
            map_dir = bkgd_wps_dir + 'energy_bin_' + str(ie) + '/'  

    # load father pixel and data
    npix = int( float(sys.argv[5]) )
    data_dir = map_dir + 'projected_maps/' 
    patch_dir = data_dir + 'map_' + str(npix) + '/'
    if map_type == 'all' or map_type == 'bkgd':
        projected_map_name = 'projected_map_dict.npz'
    elif map_type == 'ps':
        projected_map_name = 'ps_projected_map_dict' + '.npz'
    projected_map = dict(np.load(patch_dir + projected_map_name))
    data = projected_map['all_points']

    # grid and projected maps are now in two different dictionaries
    # 'outer_region_edge', which specifies the grid, is defined using the total background map for all energies
    grid_scale_deg = float(sys.argv[6])
    str_grid_scale_deg = str.format('{0:.4f}',grid_scale_deg)
    bkgd_patch_dir_all_energies = model_dir + 'bkgd/' + 'energy_bin_' + str(ie) + '/' + 'projected_maps/' + 'map_' + str(npix) + '/'
    grid_dict = dict(np.load(bkgd_patch_dir_all_energies + 'grid_dict_' + str_grid_scale_deg + '_' + '.npz'))
    mesh_bxby = grid_dict['rectangular_grid']
    grid = grid_dict['grid']

    # load scale parameter
    a_deg = float(sys.argv[8])
    a = a_deg * np.pi / 180
    arr_a = np.array([a])
    Na = len(arr_a)

    # define 2d wavelet
    wavelet_name = sys.argv[7]
    mexh = wt._2d_wavelet(wavelet_name)

    # pre-allocate wavelet coefficient map 
    N_data, N_dim = data.shape
    Ny, Nx = mesh_bxby.shape[:-1]
    coefficient_map = np.zeros((Ny, Nx, 1))

    # broadcast arrays in wavelet calculation
    buf_data = data[np.newaxis,np.newaxis,np.newaxis]
    buf_mesh_bxby = mesh_bxby[:,:,np.newaxis,np.newaxis,np.newaxis]
    buf_arr_a = arr_a[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]

    # calculate arguments of wavelet coefficients
    # buf_grid = grid[:,:,np.newaxis,np.newaxis,np.newaxis]
    buf_arr_arg_grid =  ( (buf_data - buf_mesh_bxby ) / buf_arr_a )[grid]

    # calculate wavelet coefficients
    buf_mexh_output_grid = mexh.base_fct(buf_arr_arg_grid)

    # remove two dimensions of a-array to divide mexh
    buf_arr_a_sq = np.squeeze(buf_arr_a, axis = -1)
    buf_arr_a_sq = np.squeeze(buf_arr_a_sq, axis = -1)

    # estimate wavelet coefficient by summing over datapoints
    coefficient_map_flat = np.sum(buf_mexh_output_grid, axis = -1) / buf_arr_a_sq / N_data
    coefficient_map_flat_sq = np.squeeze(coefficient_map_flat, axis = -1)

    # note that buf_arr_arg_grid is flattened relative to the shape of mesh_bxby
    # applying grid to coefficient_map automatically flattens coefficient_map to match to coefficient_map_flat_sq
    coefficient_map[grid] = coefficient_map_flat_sq

    # save coefficient estimate
    str_a_deg = str.format('{0:.5f}',a_deg)
    if map_type == 'all' or map_type == 'bkgd':
        file_name = wavelet_name + '_' + 'coefficient_map' + '_' + str_a_deg + '_' + str_grid_scale_deg
    elif map_type == 'ps':
        file_name = wavelet_name + '_' + 'ps_coefficient_map' + '_' + str_a_deg + '_' + str_grid_scale_deg 

    np.save(patch_dir + file_name, coefficient_map)

tf = timeit.default_timer()
print('Run Time (min): ' + str( (tf - ti) / 60 ) )