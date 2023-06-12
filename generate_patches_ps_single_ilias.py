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
npix = int(float(sys.argv[4]))

# inj_ids
inj_id_list = [1008,1009,1010,1015,1020,1025,1030,1035]
N_counts_list = [8,9,10,15,20,25,30,35]

for m in range(len(inj_id_list)):
    inj_id = str(inj_id_list[m])
    
    # load events from map 
    if energy_bin == 'all' or energy_bin == str(-1):
        map_dir = model_dir + ('bkgd_wps_' + inj_id + '/') 
    else:
        ie = int(float(energy_bin))
        bkgd_wps_dir = model_dir + ('bkgd_wps_' + inj_id + '/')
        map_dir = bkgd_wps_dir + 'energy_bin_' + str(ie) + '/'    

    # load events
    events = np.load(map_dir + 'ps_map.npy', allow_pickle = True) 
    l_events = events[:,0]
    b_events = events[:,1]

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
    unprojected_patches_file = map_dir + 'ps_unprojected_patches.npz'
#     if os.path.isfile(unprojected_patches_file) == True:
#         u_data = np.load(unprojected_patches_file, allow_pickle = True)
#         arr_edge_points, grouped_points_lon, grouped_points_lat, arr_c = [u_data[k] for k in u_data]
#     else:    
#         arr_edge_points = maps.healpix_edge_generator_(NSIDE = 4, step = 100)
#         grouped_points_lon, grouped_points_lat = maps.divide_data_into_groups_(lon_events, lat_events, arr_edge_points)
#         arr_c = maps.father_pixel_center_generator_(arr_edge_points)
#         unprojected_data = [arr_edge_points, grouped_points_lon, grouped_points_lat, arr_c]
#         np.savez(unprojected_patches_file, *unprojected_data)

    arr_edge_points = maps.healpix_edge_generator_(NSIDE = 4, step = 100)
    grouped_points_lon, grouped_points_lat = maps.divide_data_into_groups_(lon_events, lat_events, arr_edge_points)
    arr_c = maps.father_pixel_center_generator_(arr_edge_points)
    unprojected_data = [arr_edge_points, grouped_points_lon, grouped_points_lat, arr_c]
    np.savez(unprojected_patches_file, *unprojected_data)

    # load directories to save projected maps and plots
    data_dir = map_dir + 'projected_maps/'
    os.system("mkdir -p "+data_dir)

    plot_dir = map_dir + 'projection_plots'
    os.system("mkdir -p "+plot_dir)

    # create projected map dictionary
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
    projected_map['outer_region_points'] = maps.remove_points_from_array_(r_out,r_pix)

    # define second outer band of points outside each group (for wavelet accuracy at boundary of pixels)
    ang_scale = 4 * np.pi / 180
    lon_outmost, lat_outmost = maps.find_neighboring_points_(ang_scale, lon_events, lat_events, lon_edge, lat_edge)
    r_outmost = maps.tangent_plane_proj_(lat_outmost, lon_outmost, lat_c, lon_c)

    r_pix_and_out_redundant = np.concatenate((r_pix, r_out), axis = 0) # combine
    _, unique_indices = np.unique(r_pix_and_out_redundant, axis = 0, return_index = True) # find unique points
    r_pix_and_out = r_pix_and_out_redundant[unique_indices,:]  # remove duplicates

    r_all_redundant = np.concatenate((r_pix_and_out,r_outmost), axis = 0) # combine
    _, unique_indices = np.unique(r_all_redundant, axis = 0, return_index = True) # find unique points
    r_all = r_all_redundant[unique_indices,:]  # remove duplicates

    projected_map['all_points'] = r_all
    projected_map['outmost_region_points'] = maps.remove_points_from_array_(r_all,r_pix_and_out)

    # save file
    patch_dir = data_dir + 'map_' + str(npix) + '/'
    print(patch_dir)
    os.system("mkdir -p "+patch_dir)
    file_name = 'ps_projected_map_dict.npz'
    np.savez(patch_dir + file_name, **projected_map)

    # load background patch data to combine with ps data
    if energy_bin == 'all' or energy_bin == str(-1):
        bkgd_patch_dir = model_dir + 'bkgd/' + 'projected_maps/' + 'map_' + str(npix) + '/'
    else:
        ie = int(float(energy_bin))
        bkgd_patch_dir = model_dir + 'bkgd/' + 'energy_bin_' + str(ie) + '/' + 'projected_maps/' + 'map_' + str(npix) + '/'

    bkgd_projected_map = dict(np.load(bkgd_patch_dir + 'projected_map_dict.npz'))

    tot_projected_map = {}
    tot_projected_map['search_region_points'] = np.concatenate((bkgd_projected_map['search_region_points'], 
                                                     projected_map['search_region_points']), axis = 0)  # combine
    tot_projected_map['outer_region_points'] = np.concatenate((bkgd_projected_map['outer_region_points'], 
                                                     projected_map['outer_region_points']), axis = 0) # combine
    tot_projected_map['outmost_region_points'] = np.concatenate((bkgd_projected_map['outmost_region_points'], 
                                                     projected_map['outmost_region_points']), axis = 0) # combine
    tot_projected_map['all_points'] = np.concatenate((bkgd_projected_map['all_points'], 
                                                     projected_map['all_points']), axis = 0) # combine
    print(bkgd_projected_map['all_points'].shape)
    print(projected_map['all_points'].shape)
    print(tot_projected_map['all_points'].shape)

    # can choose to take these from either dictionary
    tot_projected_map['search_region_edge'] = bkgd_projected_map['search_region_edge']
    tot_projected_map['center_coords'] = bkgd_projected_map['center_coords']

    # ============== MODIFICATION HERE ==================
    # load outer region edge defined by all background points 
    # bkgd_patch_dir_all_energies = model_dir + 'bkgd/' + 'projected_maps/' + 'map_' + str(npix) + '/'
    # bkgd_projected_map_all_energies = dict(np.load(bkgd_patch_dir_all_energies + 'projected_map_dict.npz'))
    tot_projected_map['outer_region_edge'] = bkgd_projected_map['outer_region_edge']

    file_name = 'projected_map_dict.npz'
    np.savez(patch_dir + file_name, **tot_projected_map)

#     # plot results as a check
#     x_edge = r_edge[:,0] ; y_edge = r_edge[:,1]

#     plt.scatter(tot_projected_map['outmost_region_points'][:,0],tot_projected_map['outmost_region_points'][:,1], c = 'k', alpha = 0.3, s = 20)
#     plt.scatter(tot_projected_map['search_region_points'][:,0],tot_projected_map['search_region_points'][:,1], c = 'b', alpha = 0.3, s = 20)
#     plt.scatter(tot_projected_map['outer_region_points'][:,0],tot_projected_map['outer_region_points'][:,1],c = 'r', alpha = 0.3, s = 20)
#     plt.scatter(tot_projected_map['all_points'][:,0], tot_projected_map['all_points'][:,1], c = 'k', alpha = 0.3, s = 20)
#     plt.scatter(0,0,c = 'g', marker = '*')
#     plt.plot(x_edge,y_edge, c = 'k')
#     plt.plot(tot_projected_map['outer_region_edge'][:,0], tot_projected_map['outer_region_edge'][:,1], c = 'k', lw = 2)

#     plt.savefig(os.path.join(plot_dir, str(npix)))
#     plt.clf()