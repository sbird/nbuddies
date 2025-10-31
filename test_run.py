#############
# This is supposed to be an integration test for the full simulation. Please run this 
# file once and see if the correct movie is coming out before you push any changes 

# the correct movie for 'interesting_ICs_3body.pkl' initial conditions with 
# tot_time = 5e16, nsteps = 10, eta = 0.1 is stored in 'interesting_movie_3body.mkv' for reference
#############

import numpy as np 
from evolution2 import simulation
from ICs import generate_plummer_initial_conditions
from visualizations import movie_3D
from pint import UnitRegistry 
import pickle
import shutil
import os
import pickle as pkl

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

output_folder_data = "./data/"
output_folder_movie = "./movie_dump/"

def check_directory(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)  # Remove the folder and all its contents

    os.makedirs(folder_path)  #make new empty folder

n = 3
mass = 1.0e7
scale = 1
path_to_save_pkl_file = "./"

blackholes, masses = generate_plummer_initial_conditions(n, mass, scale)
pkl.dump({'data' : blackholes}, open(f"{path_to_save_pkl_file}/{n}body_plummer.pkl", "wb"))

initial_nbody = f'{n}body_plummer.pkl'

check_directory(output_folder_data)
simulation(initial_file = initial_nbody, output_folder = output_folder_data, tot_time = tot, nsteps = nstep, adaptive_dt = True, eta = eta)

check_directory(output_folder_movie)
movie_3D(tot_nstep_eta = f'{tot}_{nstep}_{eta}_plummer')
# saves trajectories_{tot}_{nstep}_{eta}.mkv in the current directory
# with open('BH_data_ic.pkl', 'rb') as f:
#     ics = pickle.load(f)
# print(ics)