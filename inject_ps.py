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
inj_id = sys.argv[3]
maps_dir = "/het/p4/"+username+"/gcewavelets/skysearch/data/maps/"
model_dir = maps_dir + (model + '_' + trial_id + '/')
bkgd_wps_dir = model_dir + ('bkgd_wps_' + inj_id + '/')
os.system("mkdir -p "+bkgd_wps_dir)

# load energy bins
energy_list, energy_centers = maps.generate_energy_bins_()

# produce random point source total map
if model == 'SA0':
    N_ps = 3000
    N_counts = 4
    l_list = 2 * np.pi * np.random.random(N_ps)
    b_list = ( -np.pi/2 + np.arccos( 2 * np.random.random(N_ps) - 1) ) 
    popt = np.flip([ 6.2652034 , -1.48602662,  0.05294771]) # PSF from psf_fitting notebook
elif model == 'ilias_60x60':
    N_ps = int(6000 / 12) # solid angle of 60x60 region is pi/3 (compared to 4pi)
    N_counts = 3
    l_list = np.pi / 3 * np.random.random(N_ps) - np.pi / 6
    b_list = ( -np.pi/2 + np.arccos( 1 / 2 * (2 * np.random.random( int(N_ps) ) - 1 )) )
    popt = np.flip([6.56775576, -1.58598391,  0.06022358]) # PSF from psf_fitting notebook

ps_loc = np.stack((l_list, b_list), axis = -1)
for ie in range(len(energy_list)):
    ps_dict_list = [maps.generate_pointsource_(l_list[n],b_list[n],
                                          N_counts,energy_list,popt,energy_bin=ie
                                          ,randomize_number=True) for n in range(N_ps)]
    # convert ps data dictionary to an array as maps.dict_to_array_ fct
    for n in range(N_ps):
        l_events = ps_dict_list[n]['smeared_coords'].l.rad
        b_events = ps_dict_list[n]['smeared_coords'].b.rad
        e_events = ps_dict_list[n]['energies']
        e_bin_type = ie * np.ones(len(l_events))
        if n == 0:
            ps_events_ie = np.stack((l_events, b_events, e_events, e_bin_type), axis = -1)
        else:
            ps_events_ie_one = np.stack((l_events, b_events, e_events, e_bin_type), axis = -1)
            ps_events_ie = np.concatenate((ps_events_ie,ps_events_ie_one), axis = 0)
    
    if ie == 0:
        ps_events = ps_events_ie
    else:
        ps_events = np.concatenate((ps_events,ps_events_ie), axis = 0)

# save point source map
file_name_ps_events = 'ps_map'
np.save(bkgd_wps_dir + file_name_ps_events, ps_events)

# load background map 
maps_dir = "/het/p4/"+username+"/gcewavelets/skysearch/data/maps/"
model_dir = maps_dir + (model + '_' + trial_id + '/')
bkgd_dir = model_dir + 'bkgd/'
file_name = 'map.npy'
bkgd_events = np.load(bkgd_dir + file_name, allow_pickle = True)

# combine bkgd and ps maps
events = np.concatenate((bkgd_events,ps_events), axis = 0)
np.save(bkgd_wps_dir + file_name, events)

# save maps corresponding to different energy bins
for ie in range(len(energy_list)):
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
    file_name = 'map.npy'
    np.save(map_dir_ie + file_name, events_ie)
    
# save point source positions (simple case where all point sources exist in entire spectrum)
np.save(bkgd_wps_dir + 'ps_loc', ps_loc)

# test plot
fig = plt.figure(1,figsize=(12,8))
ax = fig.add_subplot(111, projection='mollweide')

phi_events = events[:,0]
b_events = events[:,1]

phi_events[phi_events>np.pi] = phi_events[phi_events>np.pi]-2*np.pi

s = ax.hexbin(phi_events,
              b_events,gridsize=200,cmap='plasma',mincnt=1,bins='log')

plt.savefig('test_ps')
plt.show()