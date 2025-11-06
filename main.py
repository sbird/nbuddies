from src.ICs import *
from src.evolution import simulation
import os
import shutil

#name
name = "mass_segregation"

nbuddies_path = os.path.dirname(os.path.realpath(__file__))
data_path = nbuddies_path+"/data/"+name

# empty directory or creat if not exist
if os.path.exists(data_path):
    shutil.rmtree(data_path)
os.makedirs(data_path)

# Make ICs
BHs, masses = generate_plummer_initial_conditions(n_blackholes=50, initial_mass=1e7, scale=5, ratio=0.1)

pkl.dump(BHs, open(data_path+"/ICs.pkl", "wb"))

#run_sim
simulation(data_path+"/ICs.pkl", data_path, tot_time=1e17, nsteps=10, adaptive_dt=True, eta=0.1, use_tree=True)