import numpy as np
import sys, os
import healpy as hp
import _maps as maps
from tqdm import tqdm

from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage import data, img_as_float

# load local directory
username="ramirez"
local_dir = "/het/p4/"+username+"/gcewavelets/skysearch/"
maps_dir = "/het/p4/"+username+"/gcewavelets/skysearch/data/maps/"

# ---
 # Block: Specify map ids
model = 'SA0' # only available option: SA0
trial_id = str(0)
model_dir = maps_dir + (model + '_' + trial_id + '/')
energy_bin = 'all'
map_type = 'all'

# load events from map 
if energy_bin == 'all' or energy_bin == str(-1):
    if map_type == 'bkgd':
        map_dir = model_dir + 'bkgd/'
    elif map_type == 'all':
        inj_id = str(0)
        map_dir = model_dir + ('bkgd_wps_' + inj_id + '/') 
else:
    ie = int(float(energy_bin))
    if map_type == 'bkgd':
        bkgd_dir = model_dir + 'bkgd/'
        map_dir = bkgd_dir + 'energy_bin_' + str(ie) + '/'
    elif map_type == 'all':
        inj_id = str(0)
        bkgd_wps_dir = model_dir + ('bkgd_wps_' + inj_id + '/')
        map_dir = bkgd_wps_dir + 'energy_bin_' + str(ie) + '/'  

data_dir = map_dir + 'projected_maps/'
# ---

# load father pixel and data
NSIDE = 4
NPIX = hp.nside2npix(NSIDE)

r_ps_list = []
for npix in tqdm(range(NPIX)):
    patch_dir = data_dir + 'map_' + str(npix) + '/'
    
    a_deg = 0.6
    str_a_deg = str.format('{0:.5f}',a_deg)

    wavelet_name = 'mexh'
    grid_scale_deg = 0.1 
    str_grid_scale_deg = str.format('{0:.4f}',grid_scale_deg)
    file_name = wavelet_name + '_' + 'coefficient_map' + '_' + str_a_deg + '_' + str_grid_scale_deg + '.npy'
    cwt_map = np.load(patch_dir + file_name, allow_pickle = True)
    projected_map = dict(np.load(patch_dir + 'projected_map_dict.npz'))

    # decided to separate grid and projected maps into two different dictionaries
    # grid is defined using the total background map for all energies
    bkgd_patch_dir_all_energies = model_dir + 'bkgd/' + 'projected_maps/' + 'map_' + str(npix) + '/'
    grid_dict = dict(np.load(bkgd_patch_dir_all_energies + 'grid_dict_' + str_grid_scale_deg + '_' + '.npz'))
    mesh_bxby = grid_dict['rectangular_grid']
    grid = grid_dict['grid']
   
    r_out_edge = projected_map['outer_region_edge']
    x_out_edge = r_out_edge[:,0] ; y_out_edge = r_out_edge[:,1]
    bx_min, bx_max, by_min, by_max = [np.min(x_out_edge), np.max(x_out_edge),
                                      np.min(y_out_edge), np.max(y_out_edge)]
    
    scale = grid_scale_deg * np.pi / 180
    step_size = scale 
    mesh_bxby, arr_b, arr_bx_plot, arr_by_plot = maps.build_mesh_(bx_min, bx_max, by_min, by_max, step_size, step_size, return_arrays_for_plotting=True)
    
    S0 = 1
    cwt_map_thresh = np.copy(cwt_map[:,:,0])
    thresh_map = (cwt_map_thresh > S0)
    cwt_map_thresh[~thresh_map] = 0

    # single a-value 
    psf_degree = 0.4 
    psf_scale = psf_degree * np.pi / 180
    grid_scale = scale 
    N_pix_psf = int(psf_scale / grid_scale)
    
    # Comparison between image_max and im to find the coordinates of local maxima
    coordinates = peak_local_max(cwt_map_thresh, min_distance=N_pix_psf, 
                                 threshold_abs = S0)
    
    arr_bx = 0.5 * (arr_bx_plot[:-1] + arr_bx_plot[1:])
    arr_by = 0.5 * (arr_by_plot[:-1] + arr_by_plot[1:])

    nx = coordinates[:,1]
    ny = coordinates[:,0]

    pos_x = arr_bx[nx]
    pos_y = arr_by[ny]
    
    arr_edge_points = projected_map['search_region_edge']
    x_edge = arr_edge_points[:,0] ; y_edge = arr_edge_points[:,1]
    x_in, y_in = maps.find_points_inside_curve_(pos_x, pos_y, x_edge, y_edge)
    
    lon_c, lat_c = projected_map['center_coords']
    r_ps = np.array([maps.inv_tangent_plane_proj_(x_in[n], y_in[n], lon_c, lat_c) for n in range(len(x_in))])

    r_ps_list.append(r_ps)
    
np.save('ps_candidates_2', r_ps_list)