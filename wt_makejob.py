#/usr/bin/env python
#

import sys, os, time, fileinput
import numpy as np
import glob
import _maps as maps

#username
username="ramirez"

#make folder for python scripts
executedir = "wt_batch_exec"
os.system("rm -rf "+executedir)
os.system("mkdir "+executedir)

# data file for iteration

# load npix
import healpy as hp
NSIDE = 4
NPIX = hp.nside2npix(NSIDE)
arr_npix = np.arange(NPIX)

# load energy bins
## wavelet map including all energies
arr_ie = [-1] 
## wavelet map for each energy bin
# energy_list = maps.generate_energy_bins_()
# arr_ie = [ie for ie in range(len(energy_list))]

## load scales
arr_a_deg = np.array([0.6])
# step_size = 0.005
# arr_by_plot = np.arange(0,1+step_size,step_size, dtype = float)
# arr_by = 0.5 * (arr_by_plot[:-1] + arr_by_plot[1:])

# load iteration of injected ps map
N_inj = 1
arr_inj_id = np.arange(1)

# define mesh grid (see mesh_studies.ipynb for ordering)
mesh_a_deg, mesh_inj_id, mesh_ie, mesh_npix = np.meshgrid(arr_a_deg, arr_inj_id, arr_ie, arr_npix) # each output array (NxN shaped) contains x or y value at given (i,j)-th position
mesh = np.stack((mesh_npix, mesh_ie, mesh_a_deg, mesh_inj_id), axis=-1)

NPIX = len(arr_npix)
Na = len(arr_a_deg)
Nie = len(arr_ie)
arr_inputs = mesh.reshape(Na*NPIX*Nie*N_inj,4) # flatten to 2D array

# formats for each column are different; standard choice for a is 5 decimal places
# https://stackoverflow.com/questions/40030481/numpy-savetxt-save-one-column-as-int-and-the-rest-as-floats 
np.savetxt('inputs_list', arr_inputs, fmt = '%i %i %.5f %i', delimiter = ', ') # fmt specifies integer format
os.system("mv inputs_list " + executedir) # send to batch file for loading

# map ids
model = 'SA0'
trial_id = str(0)
map_type = 'all'

# wavelet-specific inputs
wavelet_name = 'mexh'
grid_scale_deg = 0.1
str_grid_scale_deg = str.format('{0:.4f}',grid_scale_deg)

#open file to contain submission commands for jdl files
dofile = open(executedir+"/do_all.src",'w')

basename = 'wt_'

# # identify directory containing results produced by main script
# RESULTS_DIR = "/het/p4/"+username+"/gcewavelets/skysearch/results/preprocessed"

# # make directory if it doesn't already exist
# os.system("mkdir -p "+RESULTS_DIR)
    
# make condor directory
condor_dir = '/het/p4/'+username+'/gcewavelets/skysearch/condor'
os.system("mkdir -p "+condor_dir)

#define the local scratch folder
localdir = "$_CONDOR_SCRATCH_DIR"
gen_command = ( 'python '+'/het/p4/'+username+
               '/gcewavelets/skysearch/code/generate_wavelet_coefficients.py')

runname = basename

# key difference: iterate over exec.sh file, not the python file itself!
execfilename = "exec"+".sh"
executefile = open(executedir+"/"+execfilename,'w')
executefile.write("#!/bin/bash\n")
executefile.write("export VO_CMS_SW_DIR=\"/cms/base/cmssoft\"\n")
executefile.write("export COIN_FULL_INDIRECT_RENDERING=1\n")
executefile.write("export SCRAM_ARCH=\"slc5_amd64_gcc434\"\n")
executefile.write("source $VO_CMS_SW_DIR/cmsset_default.sh\n")

#copy template directory to new location, and update its random number seed and run name
executefile.write("cd "+localdir+"\n")
#    executefile.write("mkdir -p results\n")
executefile.write("npix=$1\n")
executefile.write("ie=$2\n")
executefile.write("a=$3\n")
executefile.write("injid=$4\n")
executefile.write(gen_command
                  +' '+model+' '+trial_id+' '+'$ie'+' '+map_type
                  +' '+'$npix'
                  +' '+str_grid_scale_deg+' '+wavelet_name+' '+'$a'
                  +' '+'$injid'+'\n')
#    executefile.write('tar -czvf '+runname+'.tar.gz results/*\n')
#    executefile.write('cp *tar.gz '+RESULTS_DIR+'\n')
# executefile.write('cp -r * '+RESULTS_DIR+'\n')
executefile.close()
os.system("chmod u+x "+executedir+"/"+execfilename)

#write out jdl script for job submission
jdlfilename = "exec"+".jdl.base"
jdlfile = open(executedir+"/"+jdlfilename,'w')
jdlfile.write("universe = vanilla\n")
jdlfile.write("+AccountingGroup = \"group_rutgers."+username+"\"\n")
jdlfile.write("Arguments = $(npix) $(ie) $(a) $(injid)\n")
jdlfile.write("Executable = /het/p4/"+username+"/gcewavelets/skysearch/code/"+executedir+"/"+execfilename+"\n")
jdlfile.write("getenv = True\n")
jdlfile.write("should_transfer_files = NO\n")
jdlfile.write("priority = 0\n")
jdlfile.write("Output = /het/p4/"+username+"/gcewavelets/skysearch/condor/"+runname+'$(npix)'+'_'+'$(ie)'+'_'+'$(a)'+'$(injid)'+".out\n")
jdlfile.write("Error = /het/p4/"+username+"/gcewavelets/skysearch/condor/"+runname+'$(npix)'+'_'+'$(ie)'+'_'+'$(a)'+'$(injid)'+".err\n")
jdlfile.write("Log = /het/p4/"+username+"/gcewavelets/skysearch/condor/script.condor\n")
jdlfile.write("max_materialize = 200\n") # needs to be placed before queue
jdlfile.write("queue npix,ie,a,injid from inputs_list\n") # data file should not have format
jdlfile.close()

dofile.write("condor_submit "+jdlfilename+"\n")

print('Done!')

dofile.close()