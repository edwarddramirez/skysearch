from __future__ import print_function, division
import os
import sys
import pickle
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

# load energy bins
energy_list, energy_centers = maps.generate_energy_bins_()

# load background map
## load local directory
username="ramirez"
local_dir = "/het/p4/"+username+"/gcewavelets/skysearch/"

model = sys.argv[1] # only available option: SA0

# produce random SA0 skymap from the raw background files
if model == 'SA0':
    galprop_dir = local_dir + 'data/GALPROP/'
    for i in range(len(energy_list)):
        map1 = hp.read_map(galprop_dir + model + '/ics_anisotropic_healpix_57' + '_' + model + '_' + 'R12.fits', i) # Flux in [MeV^2 cm^−2 sr^−1 s^−1 MeV^−1]
        map2 = hp.read_map(galprop_dir + model + '/ics_isotropic_healpix_57' + '_' + model + '_' + 'R12.fits', i)
        map3 = hp.read_map(galprop_dir + model + '/pi0_decay_healpix_57' + '_' + model + '_' + 'R12.fits', i)
        map4 = hp.read_map(galprop_dir + model + '/synchrotron_healpix_57' + '_' + model + '_' + 'R12.fits', i)
        map5 = hp.read_map(galprop_dir + model + '/bremss_healpix_57' + '_' + model + '_' + 'R12.fits', i)
        result_map = map1 + map2 + map3 + map4 + map5
        if i==0:
            background_maps = np.zeros( (len(energy_list), len(result_map)) )
        background_maps[i] = map1+map2+map3+map4+map5
    
    # load exposure map
    # for example purposes, will use a simple overall scaling in order to create mock maps
    # NEED TO RETURN TO THIS FOR ACTUAL ANALYSIS
    exposure = 1e6
    exposure_factor = np.sum(background_maps,axis=1)/np.sum(background_maps)

    # random generation of background map
    random_map = np.einsum('ij,i->ij',background_maps,exposure_factor*exposure)

    # iterate through healpy cells, convert counts in healpy cells to individual events smeared by psf
    nside = 256
    popt = np.flip([ 6.2652034 , -1.48602662,  0.05294771]) # PSF from psf_fitting notebook
    randomized_dict = maps.healpix_counts_to_events_(random_map, energy_list, popt, nside)
    
# produce random ilias 60x60 map from Matt's file
elif model == 'ilias_60x60':
    with open('data/ilias/skymap60x60_dictionary_new.pkl', 'rb') as f:
        randomized_dict = pickle.load(f)

# convert randomized_dict to numpy array for projection code
events = maps.dict_to_array_(randomized_dict,energy_list)

# save map
trial_id = sys.argv[2]
maps_dir = "/het/p4/"+username+"/gcewavelets/skysearch/data/maps/"
model_dir = maps_dir + (model + '_' + trial_id + '/')

map_dir = model_dir + 'bkgd/'
os.system("mkdir -p "+map_dir)
file_name = 'map.npy'
np.save(map_dir + file_name, events)

# save maps corresponding to different energy bins
for ie in range(len(energy_list)):
    is_in_bin = (events[:,-1] == ie)
    events_ie = events[is_in_bin]
    
    map_dir_ie = map_dir + 'energy_bin_' + str(ie) + '/'
    os.system("mkdir -p "+map_dir_ie)
    file_name = 'map.npy'
    np.save(map_dir_ie + file_name, events_ie)