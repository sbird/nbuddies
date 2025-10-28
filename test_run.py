#############
# This is supposed to be an integration test for the full simulation. Please run this 
# file once and see if the correct movie is coming out before you push any changes 

# the correct movie for 'interesting_ICs_3body.pkl' initial conditions with 
# tot_time = 5e16, nsteps = 10, eta = 0.1 is stored in 'interesting_movie_3body.mkv' for reference
#############

import numpy as np 
from evolution2 import simulation
# from ICs import generate_initial_conditions
from visualizations import movie_3D
from pint import UnitRegistry 
import pickle

ureg = UnitRegistry()

# np.random.seed(1)
# init_data = generate_initial_conditions(3, 3, (1e7, 1e8), 5)
# with open('BH_data_ic.pkl', 'wb') as f:
#     pickle.dump({'data': init_data[0]}, f)

# the above must be changed to be compatible with the latest plummer IC stuff pushed to ICs.py  
# or its output ICs for 3 BHs from before can be used, which has been manually in 'interesting_ICs_3body.pkl' 

tot = 5e16
nstep = 10
eta = 0.1

simulation(initial_file = 'interesting_ICs_3body.pkl', output_folder = './data', tot_time = tot, nsteps = nstep, adaptive_dt = True, eta = eta)

# FIXME: any pre-existing movie_dump/ and data/ directories need to be deleted before running 
# the following function, else it will partially overwrite and may give unexpected movies. 

movie_3D(tot_nstep_eta = f'{tot}_{nstep}_{eta}')
# saves trajectories_{tot}_{nstep}_{eta}.mkv in the current directory