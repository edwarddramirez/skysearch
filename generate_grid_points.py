import numpy as np
import healpy as hp
import _maps as maps

import os, sys, time
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib as mpl

plt.style.use('default') 

# load local directory
username="ramirez"
maps_dir = "/het/p4/"+username+"/gcewavelets/skysearch/data/maps/"

# load map type (with ids corresponding to energy bin, random iteration, bkgd/ps ids)
model = sys.argv[1] # only available option: SA0
trial_id = sys.argv[2]
model_dir = maps_dir + (model + '_' + trial_id + '/')
energy_bin = sys.argv[3]
map_type = sys.argv[4]

# load events from map 
if energy_bin == 'all' or energy_bin == str(-1):
    if map_type == 'bkgd':
        map_dir = model_dir + 'bkgd/'
    elif map_type == 'all':
        inj_id = sys.argv[6]
        map_dir = model_dir + ('bkgd_wps_' + inj_id + '/') 

else:
    ie = int(float(energy_bin))
    if map_type == 'bkgd':
        bkgd_dir = model_dir + 'bkgd/'
        map_dir = bkgd_dir + 'energy_bin_' + str(ie) + '/'
    elif map_type == 'all':
        inj_id = sys.argv[6]
        bkgd_wps_dir = model_dir + ('bkgd_wps_' + inj_id + '/')
        map_dir = bkgd_wps_dir + 'energy_bin_' + str(ie) + '/'  

# load projected maps
NSIDE = 4
NPIX = hp.nside2npix(NSIDE)

# create dictionary by creating a rectangular grid and finding points that lie inside
# the boundary curve given by 'outer_region_edge'
grid_dict = {}
for npix in tqdm(range(NPIX)):
    data_dir = map_dir + 'projected_maps/' 
    patch_dir = data_dir + 'map_' + str(npix) + '/' 
    file_name = 'projected_map_dict.npz'
    projected_map = dict(np.load(patch_dir + file_name))

    # generate rectangular grid
    grid_scale_deg = float(sys.argv[5])
    str_grid_scale_deg = str.format('{0:.4f}',grid_scale_deg)
    grid_scale = grid_scale_deg * np.pi / 180
    r_out_edge = projected_map['outer_region_edge']
    x_out_edge = r_out_edge[:,0] ; y_out_edge = r_out_edge[:,1]

    # generate grid lying inside 'outer_region_edge'
    grid, mesh_bxby, arr_b, grid_flat, arr_bx_plot, arr_by_plot = maps.generate_grid_points_(x_out_edge, y_out_edge, grid_scale, True)

    grid_dict = {}
    grid_dict['grid'] = grid
    grid_dict['rectangular_grid'] = mesh_bxby
    grid_dict['rectangular_grid_points'] = arr_b
    grid_dict['grid_flat'] = grid_flat
    grid_dict['arr_bx_plot'] = arr_bx_plot
    grid_dict['arr_by_plot'] = arr_by_plot
    
    file_name = 'grid_dict_' + str_grid_scale_deg + '_' + '.npz'
    np.savez(patch_dir + file_name, **grid_dict)
    
# #     # plot to check performance 
#     plot_dir = map_dir + 'grid_points_plots_' + str_grid_scale_deg ; os.system("mkdir -p "+plot_dir)
#     Ny,Nx = mesh_bxby.shape[:-1]
#     binary_array = np.zeros((Ny,Nx))
#     binary_array[grid] = 1
    
#     plt.pcolormesh(arr_bx_plot, arr_by_plot, binary_array, cmap=plt.cm.gray, shading='flat')

#     plt.plot(x_out_edge, y_out_edge, c = 'red')
#     plt.scatter(arr_b[:,0], arr_b[:,1], s = 0.4, c = 'lightblue')
#     plt.scatter(arr_b[grid_flat,0], arr_b[grid_flat,1], c = 'r', s = 0.4)
#     plt.savefig(os.path.join(plot_dir, str(npix)))
#     plt.clf()
    
    # save