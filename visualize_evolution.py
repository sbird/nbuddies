#############
# This is supposed to be an integration test for the full simulation. Please run this 
# file once and see if the correct movie is coming out before you push any changes 

# the correct movie for 'interesting_ICs_3body.pkl' initial conditions with 
# tot_time = 5e16, nsteps = 10, eta = 0.1 is stored in 'interesting_movie_3body.mkv' for reference
#############

import numpy as np 
from src.evolution import simulation
from src.ICs import generate_plummer_initial_conditions
from src.visualizations import movie_3D
import shutil
import os
import pickle as pkl
import time
from visualizations_comparisons import movie_3D_comparison
from src.BlackHoles_Struct import BlackHole
# from binary import generate_binary_ICs

start = time.time()

def test_simulation_run(n_bh = 3, do_tree = True, do_brute = False, 
                        do_comparison = False, do_binary = False):
    
    tot = 5e16   #total time in seconds for integration
    nstep = 10  #number of steps per batch
    eta = 0.1   #adaptive time step parameter

    output_folder_data_tree = "./data_tree/"
    output_folder_data_brute = "./data_brute/"
    output_folder_movie = "./movie_dump/"

    end = []

    #check if the folder exists, if yes delete and make a new one, else just make a new one 
    def check_directory(folder_path):
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)  # Remove the folder and all its contents

        os.makedirs(folder_path)  #make new empty folder

    path_to_save_pkl_file = "./"

    if do_binary:
        custom_vals = { 'N': 2,
                'mass': np.array([1.0e7, 1.0e7]), 
                'position': np.array([[1., 0., 0.], [-1., 0., 0.]]), 
                'velocity': np.array([[0. ,3.2791 ,0.], [0. ,-3.2791 ,0.]])}
        # generate_binary_ICs(N_BH = 2, custom_vals = custom_vals, analy_sets = False)  #some problem with binary.py
        init_BH_masses = custom_vals['mass']    
        init_BH_positions = custom_vals['position']
        init_BH_velocities = custom_vals['velocity']
         ## Load the custom_vals into Class objects
        list_of_BH = []
        n = custom_vals['N']
        for i in range(n):
            BH = BlackHole( init_BH_masses[i], init_BH_positions[i], init_BH_velocities[i] )
            list_of_BH.append(BH)

        data = dict()
        data['data'] = list_of_BH
        ## Save the file
        with open('BH_data_binary.pkl', 'wb') as handle:
            pkl.dump(data, handle, protocol=pkl.HIGHEST_PROTOCOL)
        initial_values = 'BH_data_binary.pkl'
    else: 
        n = n_bh # number of black holes, usually >=2
        mass = 1.0e7  # mass of each black hole in solar mass units - same for all bodies
        scale = 1  # scale parameter "a" for Plummer model
        blackholes, _ = generate_plummer_initial_conditions(n, mass, scale)
        pkl.dump(blackholes, open(f"{path_to_save_pkl_file}/{n}body_plummer.pkl", "wb"))
        initial_values = f'{n}body_plummer.pkl'
    check_directory(output_folder_movie) #check if the movie dump folder exists

    if do_tree:
        check_directory(output_folder_data_tree) # check if the data tree folder exists
        simulation(initial_file = initial_values, output_folder = output_folder_data_tree, tot_time = tot, nsteps = nstep, adaptive_dt = True, eta = eta, use_tree= True)
        # simulation(initial_file = 'interesting_ICs_3body.pkl', output_folder = output_folder_data_tree, tot_time = tot, nsteps = nstep, adaptive_dt = True, eta = eta, use_tree= True)
        movie_3D(tot_nstep_eta = f'{n}BH_{tot}_{nstep}_{eta}_plummer_tree', brute_or_tree="/data_tree")
        end = time.time()
        print(f"Execution time for BHT: {(end - start) / 60:.2f} minutes")

    if do_brute:
        check_directory(output_folder_data_brute) # check if the data brute folder exists
        simulation(initial_file = initial_values, output_folder = output_folder_data_brute, tot_time = tot, nsteps = nstep, adaptive_dt = True, eta = eta, use_tree= False)
        # simulation(initial_file = 'interesting_ICs_3body.pkl', output_folder = output_folder_data_brute, tot_time = tot, nsteps = nstep, adaptive_dt = True, eta = eta, use_tree= False)
        movie_3D(tot_nstep_eta = f'{n}BH_{tot}_{nstep}_{eta}_plummer_brute', brute_or_tree="/data_brute")
        end = time.time()
        print(f"Execution time for brute force: {(end - start) / 60:.2f} minutes")

    if do_comparison:
        movie_3D_comparison(tot_nstep_eta = f'{n}BH_{tot}_{nstep}_{eta}_plummer_comparison')
        end = time.time()
        print(f"Execution time for comparison: {(end - start) / 60:.2f} minutes")

# saves trajectories_{tot}_{nstep}_{eta}.mkv in the current directory

test_simulation_run(n_bh=2, do_tree=True, do_brute=False, do_comparison=False, do_binary=True)

#--------------------------------------------------------------------------------#
#not needed anymore
# np.random.seed(1)
# init_data = generate_initial_conditions(3, 3, (1e7, 1e8), 5)
# with open('BH_data_ic.pkl', 'wb') as f:
#     pickle.dump({'data': init_data[0]}, f)

# the above must be changed to be compatible with the latest plummer IC stuff pushed to ICs.py  
# or its output ICs for 3 BHs from before can be used, which has been manually in 'interesting_ICs_3body.pkl' 
