import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import shutil

nbuddies_path = (os.path.dirname(os.path.realpath(__file__)))

def movie_3D_comparison(sim_name : str, tail_length: int = 10, tot_nstep_eta=None):
    """
    Loads both data_tree and data_brute simulation data and makes a combined movie
    showing both trajectories together in 3D space with tails behind them.
    
    Blue - BHT (Barnes-Hut Tree)
    Red  - BF (Brute Force)

    Parameters
    ----------
    tail_length : int, default 10
        Length of tail trailing behind points.
    tot_nstep_eta: str
        Used to dynamically save the resulting movie with info about 
        total time (sec), num of timesteps per batch, eta for adaptive timestep computation.
    """

    # set up
    if os.path.exists(nbuddies_path+"/movie_dump_comparison/"+sim_name): # check if dir exists
        shutil.rmtree(nbuddies_path+"/movie_dump_comparison/"+sim_name) #pruge old images  
    os.makedirs(nbuddies_path+"/movie_dump_comparison/"+sim_name) # create dir path

    # get info from sim end
    last_batch_tree = _find_last_batch_num(sim_name, "_tree")
    last_batch_brute = _find_last_batch_num(sim_name, "_brute")
    last_batch_num = min(last_batch_tree, last_batch_brute)  # ensure equal frame count

    # Load last batch and initialize data structures
    with open(nbuddies_path + '/data/' + sim_name + "_tree" + f"/data_batch{last_batch_num}.pkl", 'rb') as file:
        data_tree = pickle.load(file)['data'][0]
    with open(nbuddies_path + '/data/' + sim_name + "_brute" + f"/data_batch{last_batch_num}.pkl", 'rb') as file:
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
    with open(nbuddies_path + '/data/' + sim_name + "_tree/data_batch0.pkl", 'rb') as file:
        init_tree = pickle.load(file)['data'][0]
    with open(nbuddies_path  + '/data/' + sim_name + "_brute/data_batch0.pkl", 'rb') as file:
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
        with open(nbuddies_path +  '/data/' + sim_name + f"_tree/data_batch{i}.pkl", 'rb') as file:
            data_tree = pickle.load(file)['data'][0]
        with open(nbuddies_path +  '/data/' + sim_name + f"_brute/data_batch{i}.pkl", 'rb') as file:
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
        plt.savefig(nbuddies_path + "/movie_dump_comparison/" +sim_name+ f"/trajectories_{i}.png",
                    dpi=300, bbox_inches='tight')
        plt.close()

    _recompile_movie_3D_compare(sim_name)


def _find_last_batch_num(sim_name, brute_or_tree) -> int:
    """
    Finds number of last batch file saved for either data_tree or data_brute
    """
    i = 0
    while os.path.exists(nbuddies_path + '/data/' + sim_name  + brute_or_tree + f"/data_batch{i}.pkl"):
        i += 1
    return i - 1


def _recompile_movie_3D_compare(sim_name):
    """
    Deletes old comparison movie if it exists, then recreates it by compiling the PNGs in movie_dump_comparison
    """
    
    if not os.path.exists(nbuddies_path+"/visuals/"+sim_name):
        os.makedirs(nbuddies_path+"/visuals/"+sim_name)

    if os.path.exists(nbuddies_path + "/visuals/" + sim_name + f"/trajectories.mkv"): # checks if path exists
        os.remove(nbuddies_path + "/visuals/" + sim_name + f"/trajectories.mkv") # if it does, remove path to old movie file
    os.system("ffmpeg -framerate 12 -start_number 0 -i " + nbuddies_path + "/movie_dump_comparison/"+ sim_name +"/trajectories_%01d.png -q:v 0 " + nbuddies_path + "/visuals/" + sim_name + f"/trajectories.mkv") # recreate movie

