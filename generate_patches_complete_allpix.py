import numpy as np
import healpy as hp
import _maps as maps

import sys, os, time
from tqdm import tqdm

import matplotlib.pyplot as plt
plt.style.use('default') 

# load energy bins
energy_list, energy_centers = maps.generate_energy_bins_()

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
if energy_bin == 'all':
    if map_type == 'bkgd':
        map_dir = model_dir + 'bkgd/'
        events = np.load(map_dir + 'map.npy', allow_pickle = True)
    elif map_type == 'all':
        inj_id = sys.argv[5]
        map_dir = model_dir + ('bkgd_wps_' + inj_id + '/')
        events = np.load(map_dir + 'map.npy', allow_pickle = True)  

else:
    ie = int(float(energy_bin))
    if map_type == 'bkgd':
        bkgd_dir = model_dir + 'bkgd/'
        map_dir = bkgd_dir + 'energy_bin_' + str(ie) + '/'
        events = np.load(map_dir + 'map.npy', allow_pickle = True)
    elif map_type == 'all':
        inj_id = sys.argv[5]
        bkgd_wps_dir = model_dir + ('bkgd_wps_' + inj_id + '/')
        map_dir = bkgd_wps_dir + 'energy_bin_' + str(ie) + '/'
        events = np.load(map_dir + 'map.npy', allow_pickle = True)     
  
# load events
l_events = events[:,0]
b_events = events[:,1]

print(l_events.shape)

# after loading the data, we need our angular coordinates to be given by
# longitude ([0,2\pi]) and latitude (-\pi, \pi)
phi_events = l_events.copy()
phi_events[phi_events>np.pi] = phi_events[phi_events>np.pi]-2*np.pi

lon_events = phi_events + np.pi
lat_events = b_events

# load healpix pixel edges, divide data into groups, calculate center of pixels for projection
NSIDE = 4
NPIX = hp.nside2npix(NSIDE)

# group points into patches (before projection)
unprojected_patches_file = map_dir + 'unprojected_patches.npz'
if os.path.isfile(unprojected_patches_file) == True:
    u_data = np.load(unprojected_patches_file, allow_pickle = True)
    arr_edge_points, grouped_points_lon, grouped_points_lat, arr_c = [u_data[k] for k in u_data]
else:    
    arr_edge_points = maps.healpix_edge_generator_(NSIDE = 4, step = 100)
    grouped_points_lon, grouped_points_lat = maps.divide_data_into_groups_(lon_events, lat_events, arr_edge_points)
    arr_c = maps.father_pixel_center_generator_(arr_edge_points)
    unprojected_data = [arr_edge_points, grouped_points_lon, grouped_points_lat, arr_c]
    np.savez(unprojected_patches_file, *unprojected_data)

data_dir = map_dir + 'projected_maps/'
os.system("mkdir -p "+data_dir)

plot_dir = map_dir + 'projection_plots'
os.system("mkdir -p "+plot_dir)
    
for npix in tqdm(range(NPIX)):
    projected_map = {}
    
    # load and save coords of center of projected map wrt original spherical coords
    lon_c = arr_c[npix,0] ; lat_c = arr_c[npix,1]
    projected_map['center_coords'] = arr_c[npix,:]
    
    # project each group of points into their respective tangent plane
    lon_pix = grouped_points_lon[npix]
    lat_pix = grouped_points_lat[npix]

    lon_edge = arr_edge_points[npix, :, 0]
    lat_edge = arr_edge_points[npix, :, 1]
    
    r_pix = maps.tangent_plane_proj_(lat_pix, lon_pix, lat_c, lon_c)
    r_edge = maps.tangent_plane_proj_(lat_edge, lon_edge, lat_c, lon_c)
    
    projected_map['search_region_points'] = r_pix
    projected_map['search_region_edge'] = r_edge
    
    # generate first outer band of points outside each group 
    ang_scale = 2 * np.pi / 180
    lon_out, lat_out = maps.find_neighboring_points_(ang_scale, lon_events, lat_events, lon_edge, lat_edge)
    r_out = maps.tangent_plane_proj_(lat_out, lon_out, lat_c, lon_c)

#     projected_map['outer_region_points'] = r_out
    
    x_out = r_out[:,0] ; y_out = r_out[:,1]
    outer_edge = maps.generate_edge_of_point_set_(x_out, y_out)
    x_out_edge = outer_edge[:,0] ; y_out_edge = outer_edge[:,1]
    r_out_edge = np.stack((x_out_edge,y_out_edge), axis = -1)
    
    projected_map['outer_region_edge'] = r_out_edge
    
    # define second outer band of points outside each group
    ang_scale = 4 * np.pi / 180
    lon_outmost, lat_outmost = maps.find_neighboring_points_(ang_scale, lon_events, lat_events, lon_edge, lat_edge)
    r_outmost = maps.tangent_plane_proj_(lat_outmost, lon_outmost, lat_c, lon_c)
    
#     projected_map['outmost_region_points'] = r_outmost
    
    r_pix_and_out_redundant = np.concatenate((r_pix, r_out), axis = 0)
    _, unique_indices = np.unique(r_pix_and_out_redundant, axis = 0, return_index = True)
    r_pix_and_out = r_pix_and_out_redundant[unique_indices,:]
    projected_map['search_and_outer_region_points'] = r_pix_and_out
    projected_map['outer_region_points'] = maps.remove_points_from_array_(r_out,r_pix)
    
    print(r_pix.shape)
    print(r_out.shape)
    print(r_pix_and_out_redundant.shape)
    print(projected_map['outer_region_points'].shape)
    print(r_pix_and_out[r_pix_and_out==r_pix].shape)
    
    r_all_redundant = np.concatenate((r_pix_and_out, r_outmost), axis = 0)
    _, unique_indices = np.unique(r_all_redundant, axis = 0, return_index = True)
    r_all = r_all_redundant[unique_indices,:]
    projected_map['all_points'] = r_all
    projected_map['outmost_region_points'] = maps.remove_points_from_array_(r_all,r_pix_and_out)
    print('-----')
    print(r_pix_and_out.shape)
    print(projected_map['outmost_region_points'].shape)
    print(r_all.shape)
    
    r_all_alt_piece = np.concatenate((r_pix,r_out), axis = 0)
    r_all_alt_red = np.concatenate((r_all_alt_piece,r_outmost), axis = 0)
    _, unique_indices = np.unique(r_all_alt_red, axis = 0, return_index = True)
    r_all_alt = r_all_alt_red[unique_indices,:]
    print(r_all_alt.shape)
    print(np.array_equal(np.sort(r_all.flat), np.sort(r_all_alt.flat)))
    
    patch_dir = data_dir + 'map_' + str(npix) + '/'
    os.system("mkdir -p "+patch_dir)
    file_name = 'projected_map_dict.npz'
    np.savez(patch_dir + file_name, **projected_map)
    
#     x_edge = r_edge[:,0] ; y_edge = r_edge[:,1]
#     x_pix = r_pix[:,0] ; y_pix = r_pix[:,1]
#     x_outmost = r_outmost[:,0] ; y_outmost = r_outmost[:,1]

#     x_pix_and_out = r_pix_and_out[:,0] ; y_pix_and_out = r_pix_and_out[:,1]
#     plt.scatter(projected_map['outmost_region_points'][:,0],projected_map['outmost_region_points'][:,1], c = 'k', alpha = 0.3, s = 20)
#     plt.scatter(x_pix,y_pix, c = 'b', alpha = 0.3, s = 20)
#     plt.scatter(projected_map['outer_region_points'][:,0],projected_map['outer_region_points'][:,1],c = 'r', alpha = 0.3, s = 20)
#     plt.scatter(x_pix_and_out,y_pix_and_out, c = 'r', alpha = 0.3, s = 20)
# #     plt.scatter(r_all[:,0], r_all[:,1], c = 'k', alpha = 0.3, s = 20)
#     plt.scatter(0,0,c = 'g', marker = '*')
#     plt.plot(x_edge,y_edge, c = 'k')
#     plt.plot(x_out_edge, y_out_edge, c = 'k', lw = 2)
    
#     plt.savefig(os.path.join(plot_dir, str(npix)))
#     plt.clf()