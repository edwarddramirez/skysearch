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

# load father pixels and edge
NSIDE = 4
NPIX = hp.nside2npix(NSIDE)
arr_edge_points = maps.healpix_edge_generator_(NSIDE = 4, step = 100)

# load point source locations
events_loc = np.load(map_dir + 'ps_loc.npy', allow_pickle = True)
l_events_loc = events_loc[:,0]
b_events_loc = events_loc[:,1]

phi_events_loc = l_events_loc.copy()
phi_events_loc[phi_events_loc>np.pi] = phi_events_loc[phi_events_loc>np.pi]-2*np.pi

lon_events_loc = phi_events_loc + np.pi
lat_events_loc = b_events_loc

## group points into father pixels
grouped_loc_lon, grouped_loc_lat = maps.divide_data_into_groups_(lon_events_loc, lat_events_loc, arr_edge_points)

# load S0_list
S0_list = np.arange(0.001,1+0.001,0.001)
N_S0 = len(S0_list)

for npix in tqdm(range(NPIX)):
    # load files within father pixel
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
    Ny,Nx = mesh_bxby.shape[:-1]
    
    # obtain grid within only the father pixel
    arr_edge_points = projected_map['search_region_edge']
    x_edge = arr_edge_points[:,0] ; y_edge = arr_edge_points[:,1]
    
    grid_pix_flat = maps.find_points_inside_curve_(arr_b[:,0], arr_b[:,1], x_edge, y_edge, return_grid=True) # 2D array
    grid_pix = grid_pix_flat.reshape((Ny,Nx))
    grid_points = arr_b[grid_pix_flat,:]
    N_grid = grid_points.shape[0]

    # pre-allocate arrays of positive and negative detections
    arr_fp = np.zeros((N_S0))
    arr_tp = np.zeros((N_S0))
    arr_fn = np.zeros((N_S0))
    arr_tn = np.zeros((N_S0))

    # load psf degree and grid scale
    psf_degree = 0.4    # dependent on father pixel (should use maps.psf_ fct)
    psf_scale = psf_degree * np.pi / 180
    grid_scale = 0.1 * np.pi / 180
    N_pix_psf = int(3*psf_scale / grid_scale)
    
    # load point source locations
    lon_events_loc_npix = grouped_loc_lon[npix]
    lat_events_loc_npix = grouped_loc_lat[npix]

    # project point source locations to father pixel
    lon_c, lat_c = projected_map['center_coords']
    r_loc = np.array(maps.tangent_plane_proj_(lat_events_loc_npix, lon_events_loc_npix, lat_c, lon_c))
    N_ps = r_loc.shape[0]

    for n_S0 in range(N_S0):
        S0 = S0_list[n_S0]

        # threshold the wavelet map
        cwt_map_thresh = np.copy(cwt_map[:,:,0])
        thresh_map = (cwt_map_thresh > S0)
        cwt_map_thresh[~thresh_map] = 0

        # identify point source finders with peak_local_max
        coordinates = peak_local_max(cwt_map_thresh, min_distance=N_pix_psf, 
                                 threshold_abs = S0)

        # obtain positions of detections that are within the og father pixel
        nx = coordinates[:,1]
        ny = coordinates[:,0]

        arr_bx = 0.5 * (arr_bx_plot[:-1] + arr_bx_plot[1:])
        arr_by = 0.5 * (arr_by_plot[:-1] + arr_by_plot[1:])
        pos_x = arr_bx[nx]
        pos_y = arr_by[ny]

        x_in, y_in = maps.find_points_inside_curve_(pos_x, pos_y, x_edge, y_edge)
        detection_pos = np.stack((x_in, y_in), axis = -1)
        N_detect = len(x_in)

        # calculate minimum distances of detection grid points from point source
        relative_positions = r_loc[:,np.newaxis,:] - detection_pos
        distances_from_ps = np.linalg.norm(relative_positions, axis = -1)
        min_rel_pos = np.min(distances_from_ps, axis = 0)
        
        # threshold for grid point to be associated with point source (within \sigma_psf and \sigma_grid)
        thresh_dist = (min_rel_pos < np.sqrt(psf_scale**2. + grid_scale**2.) )
        min_rel_pos_thresh = min_rel_pos[thresh_dist] 

        # calculate positives and negatives
        tp = len(min_rel_pos_thresh) 
        fp = len(min_rel_pos) - len(min_rel_pos_thresh)
        fn = N_ps - tp
        tn = N_grid - tp - fp - fn

        # store into array
        arr_tp[n_S0] = tp
        arr_fp[n_S0] = fp
        arr_fn[n_S0] = fn
        arr_tn[n_S0] = tn
        
    # calculate tprs and fprs
    arr_tpr = arr_tp / (arr_tp + arr_fn)
    arr_tpr[np.isnan(arr_tpr)] = 0
    arr_fpr = arr_fp / (arr_fp + arr_tn)
    arr_fpr[np.isnan(arr_fpr)] = 0
    
    # save roc curve data
    str_a_deg = str.format('{0:.5f}',a_deg)
    roc_file_name = 'roc_' + str_a_deg 
    np.save(patch_dir + roc_file_name, [arr_tp, arr_fp, arr_fn, arr_tn])