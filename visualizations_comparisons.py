import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import shutil
from src.ICs import *
import pickle as pkl
from src.evolution import *
from pint import UnitRegistry

nbuddies_path = os.path.dirname(os.path.realpath(__file__))

def movie_3D_comparison(sim_name : str, tail_length: int = 10):
    """
    Loads both data_tree and data_brute simulation data and makes a combined movie
    showing both trajectories together in 3D space with tails behind them.
    
    Blue - BHT (Barnes-Hut Tree)
    Red  - BF (Brute Force)

    Parameters
    ----------
    tail_length : int, default 10
        Length of tail trailing behind points.
    """

    # set up
    if os.path.exists(nbuddies_path+"/addtnl_checks/movie_dump_comparison/"+sim_name): # check if dir exists
        shutil.rmtree(nbuddies_path+"/addtnl_checks/movie_dump_comparison/"+sim_name) #pruge old images  
    os.makedirs(nbuddies_path+"/addtnl_checks/movie_dump_comparison/"+sim_name) # create dir path

    # get info from sim end
    last_batch_tree = _find_last_batch_num(sim_name, "/tree")
    last_batch_brute = _find_last_batch_num(sim_name, "/brute")
    last_batch_num = min(last_batch_tree, last_batch_brute)  # ensure equal frame count

    # Load last batch and initialize data structures
    with open(nbuddies_path + '/addtnl_checks/' + sim_name + "/tree" + f"/data_batch{last_batch_num}.pkl", 'rb') as file:
        data_tree = pickle.load(file)['data'][0]
    with open(nbuddies_path + '/addtnl_checks/' + sim_name + "/brute" + f"/data_batch{last_batch_num}.pkl", 'rb') as file:
        data_brute = pickle.load(file)['data'][0]
    N_tree = len(data_tree)
    N_brute = len(data_brute)

    # Calculate the maximum plot range based on particle positions
    max_range = 0
    for dataset in [data_tree, data_brute]:
        for obj in dataset:
            if np.linalg.norm(obj.position) > max_range:
                max_range = np.linalg.norm(obj.position)
    max_range *= 2  # add buffer

    # get info from sim start
    with open(nbuddies_path + '/addtnl_checks/' + sim_name + "/tree/data_batch0.pkl", 'rb') as file:
        init_tree = pickle.load(file)['data'][0]
    with open(nbuddies_path  + '/addtnl_checks/' + sim_name + "/brute/data_batch0.pkl", 'rb') as file:
        init_brute = pickle.load(file)['data'][0]

    # Create 3D arrays to store tail positions
    plotting_tree = np.zeros([N_tree, 3, tail_length])
    plotting_brute = np.zeros([N_brute, 3, tail_length])
    for n in range(N_tree):
        for t in range(tail_length):
            plotting_tree[n, :, t] = init_tree[n].position
    for n in range(N_brute):
        for t in range(tail_length):
            plotting_brute[n, :, t] = init_brute[n].position

    # generating movie frames
    for i in range(last_batch_num + 1):

        # slide data window forward
        for j in range(tail_length - 1):
            plotting_tree[:, :, j] = plotting_tree[:, :, j + 1]
            plotting_brute[:, :, j] = plotting_brute[:, :, j + 1]

        # Load current frame data
        with open(nbuddies_path +  '/addtnl_checks/' + sim_name + f"/tree/data_batch{i}.pkl", 'rb') as file:
            data_tree = pickle.load(file)['data'][0]
        with open(nbuddies_path +  '/addtnl_checks/' + sim_name + f"/brute/data_batch{i}.pkl", 'rb') as file:
            data_brute = pickle.load(file)['data'][0]

        # Update tails
        for n in range(N_tree):
            plotting_tree[n, :, -1] = data_tree[n].position
        for n in range(N_brute):
            plotting_brute[n, :, -1] = data_brute[n].position

        # plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # tree (blue)
        for n in range(N_tree):
            for t in range(tail_length - 1):
                alpha = t / (tail_length - 2)
                ax.plot(plotting_tree[n, 0, t:t + 2],
                        plotting_tree[n, 1, t:t + 2],
                        plotting_tree[n, 2, t:t + 2],
                        'b-', alpha=alpha)
            ax.scatter(plotting_tree[n, 0, -1],
                       plotting_tree[n, 1, -1],
                       plotting_tree[n, 2, -1],
                       c="blue", s=100, marker="o")

        # brute (red)
        for n in range(N_brute):
            for t in range(tail_length - 1):
                alpha = t / (tail_length - 2)
                ax.plot(plotting_brute[n, 0, t:t + 2],
                        plotting_brute[n, 1, t:t + 2],
                        plotting_brute[n, 2, t:t + 2],
                        'r-', alpha=alpha)
            ax.scatter(plotting_brute[n, 0, -1],
                       plotting_brute[n, 1, -1],
                       plotting_brute[n, 2, -1],
                       c="red", s=100, marker="o")

        # Set plot labels and limits
        ax.set_xlabel('X [kpc]')
        ax.set_ylabel('Y [kpc]')
        ax.set_zlabel('Z [kpc]')
        ax.set_title('Black Hole Trajectories: BHT (Blue) vs BF (Red)')

        ax.set_xlim(-max_range / 2, max_range / 2)
        ax.set_ylim(-max_range / 2, max_range / 2)
        ax.set_zlim(-max_range / 2, max_range / 2)

        # Add legend
        blue_proxy = plt.Line2D([0], [0], linestyle='-', color='blue', label='BHT')
        red_proxy = plt.Line2D([0], [0], linestyle='-', color='red', label='BF')
        ax.legend(handles=[blue_proxy, red_proxy], loc='upper right')

        plt.tight_layout()
        plt.savefig(nbuddies_path + "/addtnl_checks/movie_dump_comparison/" +sim_name+ f"/trajectories_{i}.png",
                    dpi=300, bbox_inches='tight')
        plt.close()

    _recompile_movie_3D_compare(sim_name)


def _find_last_batch_num(sim_name, brute_or_tree) -> int:
    """
    Finds number of last batch file saved for either data_tree or data_brute
    """
    i = 0
    while os.path.exists(nbuddies_path + '/addtnl_checks/' + sim_name  + brute_or_tree + f"/data_batch{i}.pkl"):
        i += 1
    return i - 1


def _recompile_movie_3D_compare(sim_name):
    """
    Deletes old comparison movie if it exists, then recreates it by compiling the PNGs in movie_dump_comparison
    """
    
    if not os.path.exists(nbuddies_path+'/addtnl_checks/' + sim_name+"/visuals/"):
        os.makedirs(nbuddies_path+ '/addtnl_checks/' +sim_name+"/visuals/")

    if os.path.exists(nbuddies_path + sim_name +"/visuals/" + f"/trajectories.mkv"): # checks if path exists
        os.remove(nbuddies_path + sim_name + "/visuals/" + f"/trajectories.mkv") # if it does, remove path to old movie file
    os.system("ffmpeg -framerate 12 -start_number 0 -i " + nbuddies_path + "/addtnl_checks/movie_dump_comparison/"+ sim_name +"/trajectories_%01d.png -q:v 0 " + nbuddies_path + '/addtnl_checks/' + sim_name +'/visuals' + f"/trajectories.mkv") # recreate movie

if __name__ == "__main__":

    ureg = UnitRegistry()

    #define the parameters
    sim_name = "test_binary"             #enter desired name of simulation
    data_path = nbuddies_path + "/addtnl_checks/" + sim_name
    sim_time = 5e16 * ureg('second')
    nsteps = 10
    adap_dt = True
    eta = 0.1

    def check_directory(folder_path):
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)  # Remove the folder and all its contents

        os.makedirs(folder_path)  #make new empty folder
    
    check_directory(data_path)

    #generate ICs
    
    #choose which model you want to implement - comment out the other one

    # #plummer sphere model
    # np.random.seed(43)
    # n = 3              # number of black holes
    # mass = 1e6          # solar masses per BH
    # m1_ratio = 0.00      # mass ratio between two types of black holes
    # scale = 1           # scale (a value)
    # BHs, _ = generate_plummer_initial_conditions(n, mass, scale, m1_ratio)

    #binary model
    custom_vals = {
            'N': 2,
            'mass': np.array([1.0e7, 1.0e7]),
            'position': np.array([[1., 0., 0.], [-1., 0., 0.]]),
            'velocity': np.array([[0., 3.2791, 0.], [0., -3.2791, 0.]])
        }
    BHs, _ = generate_binary_ICs(N_BH=2, custom_vals=custom_vals)

    pkl.dump(BHs, open(data_path+"/ICs.pkl", "wb"))

    #call simulation for brute and tree methods 
    simulation(data_path+"/ICs.pkl", data_path+"/tree", tot_time=sim_time.to('second').magnitude, nsteps= nsteps, 
        adaptive_dt= adap_dt , eta= eta, use_tree=True)
    simulation(data_path+"/ICs.pkl", data_path+"/brute", tot_time=sim_time.to('second').magnitude, nsteps= nsteps, 
        adaptive_dt= adap_dt , eta= eta, use_tree=False)

    # create movie
    movie_3D_comparison(sim_name)