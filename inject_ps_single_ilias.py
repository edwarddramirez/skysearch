from __future__ import print_function, division
import os
import sys
import timeit as timeit
from tqdm import tqdm

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import math

import healpy as hp
import astropy
from astropy import coordinates as coords
from astropy.coordinates import SkyCoord
from astropy.coordinates import Galactic
from astropy import units as u

import _maps as maps

# load local directory
username="ramirez"
local_dir = "/het/p4/"+username+"/gcewavelets/skysearch/"

model = sys.argv[1]
trial_id = sys.argv[2]
maps_dir = "/het/p4/"+username+"/gcewavelets/skysearch/data/maps/"
model_dir = maps_dir + (model + '_' + trial_id + '/')

inj_id_list = [1008,1009,1010,1015,1020,1025,1030,1035]
N_counts_list = [8,9,10,15,20,25,30,35]

# load father pixels and edge
NSIDE = 4
NPIX = hp.nside2npix(NSIDE)
arr_edge_points = maps.healpix_edge_generator_(NSIDE = 4, step = 100)

npix = 112
lon_edge = arr_edge_points[npix,:,0] ; lat_edge = arr_edge_points[npix,:,1]
l_edge = lon_edge - np.pi ; b_edge = lat_edge

l_edge_min = np.min(l_edge) ; l_edge_max = np.max(l_edge) 
b_edge_min = np.min(b_edge) ; b_edge_max = np.max(b_edge)

delta_l = l_edge_max - l_edge_min
sin_b_min = np.sin(b_edge_max)
delta_sin_b = - np.sin(b_edge_max) + np.sin(b_edge_min)

energy_list, energy_centers = maps.generate_energy_bins_()
ie = 16
# load background map 
bkgd_dir = model_dir + 'bkgd/'
map_dir_ie = bkgd_dir + 'energy_bin_' + str(ie) + '/'
file_name = 'map.npy'
bkgd_events = np.load(map_dir_ie + file_name, allow_pickle = True)

N_ps = 500
popt = np.flip([6.56775576, -1.58598391,  0.06022358]) # PSF from psf_fitting notebook

for m in tqdm(range(len(N_counts_list))):
    N_counts = N_counts_list[m]
    ps_dict_list, ps_loc = maps.generate_point_sources_inside_curve_(N_ps,N_counts,npix,lon_edge,lat_edge,energy_list,popt,
                                                             ie)
    
    # convert ps data dictionary to an array as maps.dict_to_array_ fct
    for n in range(N_ps):
        l_events = ps_dict_list[n]['smeared_coords'].l.rad
        b_events = ps_dict_list[n]['smeared_coords'].b.rad
        e_events = ps_dict_list[n]['energies']
        e_bin_type = ie * np.ones(len(l_events))
        if n == 0:
            ps_events = np.stack((l_events, b_events, e_events, e_bin_type), axis = -1)
        else:
            ps_events_one = np.stack((l_events, b_events, e_events, e_bin_type), axis = -1)
            ps_events = np.concatenate((ps_events,ps_events_one), axis = 0)

    # save point source map
    inj_id = str(inj_id_list[m])
    bkgd_wps_dir = model_dir + ('bkgd_wps_' + inj_id + '/')
    os.system("mkdir -p "+bkgd_wps_dir)
    file_name_ps_events = 'ps_map'
    np.save(bkgd_wps_dir + file_name_ps_events, ps_events)

    # combine bkgd and ps maps
    events = np.concatenate((bkgd_events,ps_events), axis = 0)
    np.save(bkgd_wps_dir + file_name, events)

    # save maps corresponding to different energy bins
    # directory
    map_dir_ie = bkgd_wps_dir + 'energy_bin_' + str(ie) + '/'
    os.system("mkdir -p "+map_dir_ie)
    # ps map
    is_in_bin = (ps_events[:,-1] == ie)
    ps_events_ie = ps_events[is_in_bin]
    file_name = 'ps_map.npy'
    np.save(map_dir_ie + file_name, ps_events_ie)
    # bkgd plus ps map
    is_in_bin = (events[:,-1] == ie)
    events_ie = events[is_in_bin]
    if ie == 16:
        print(events_ie.shape)
    file_name = 'map.npy'
    np.save(map_dir_ie + file_name, events_ie)
    
    # save point source positions (simple case where all point sources exist in entire spectrum)
    np.save(bkgd_wps_dir + 'ps_loc', ps_loc)

# # test plot
# fig = plt.figure(1,figsize=(12,8))
# ax = fig.add_subplot(111, projection='mollweide')

# phi_events = events[:,0]
# b_events = events[:,1]

# phi_events[phi_events>np.pi] = phi_events[phi_events>np.pi]-2*np.pi

# s = ax.hexbin(phi_events,
#               b_events,gridsize=200,cmap='plasma',mincnt=1,bins='log')

# plt.savefig('test_ps')
# plt.show()