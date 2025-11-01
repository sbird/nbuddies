import os
import pickle
import matplotlib.pyplot as plt
import numpy as np


nbuddies_path = os.path.dirname(os.path.realpath(__file__))

def movie_3D(tail_length: int = 10, tot_nstep_eta = None, brute_or_tree = "/data_tree"):
    """
    Loads data and makes movie of motion in 3D space with tails behind them

    Parameters
    ----------
    tail_length : int, default 10
        length of tail trailing behind points
    tot_nstep_eta: str, used to dynamically save the resulting movies with info about 
                    total time (sec), num of timesteps per batch, eta for adaptive timestep computation
    brute_or_tree : str, either "/data_brute" or "/data_tree" depending on which simulation data to use
                    default "/data_tree"
    """
    
    #set up
    # Creating output folder for animation frames if it doesn't exist
    if not os.path.exists(nbuddies_path+"/movie_dump"): # check if dir exists
        os.makedirs(nbuddies_path+"/movie_dump") # if not, create dir path
    
    #getting info from sim end
    last_batch_num = _find_last_batch_num(brute_or_tree) # find number corresponding to last data batch number
    
    # Load last batch and initialize data structures
    with open(nbuddies_path + brute_or_tree + f"/data_batch{last_batch_num}.pkl", 'rb') as file: # open dir of batches
        data = pickle.load(file)['data'][0] # load pk data files with last batch number
    N = len(data) # length of data files

    #Calculate the maximum plot range based on particle positions
    max_range = 0
    for n in range(N):
        if np.linalg.norm(data[n].position) > max_range: # if Euclidian distance is greater than max_range
            max_range = np.linalg.norm(data[n].position) # set max_range to Euclidian distance
    max_range *= 2 # add buffer by increasing max_range by 25%

    #getting info from sim start
    with open(nbuddies_path + brute_or_tree + "/data_batch0.pkl", 'rb') as file:
        init_data = pickle.load(file)['data'][0]
    #Create 3D array to store tail positions
    plotting_data = np.zeros([N, 3, tail_length]) # instantiate array of zeros with dimensions N x 3 x tail_length
    for n in range(N):
        for t in range(tail_length):
            plotting_data[n, :, t] = init_data[n].position # append positions to plotting_data
    
    #generating movie frames
    #Loop through batch to make frames
    for i in range(last_batch_num + 1):
        
        #slide data window forward
        for j in range(tail_length - 1):
            plotting_data[:,:,j] = plotting_data[:,:,j+1] # move data window forward by 1
        #Load current frame data
        with open(nbuddies_path + brute_or_tree + f"/data_batch{i}.pkl", 'rb') as file: # open dir to pk files
            data = pickle.load(file)['data'][0] # load pk files
        #Update tail with current particle positions
        for n in range(N):
            plotting_data[n, :, -1] = data[n].position

        #plot
        fig = plt.figure() # instantiate figure
        ax = fig.add_subplot(111, projection='3d') # create 3D subplot

        for n in range(N):
            #plotting tails
            for t in range(tail_length - 1):
                alpha = t / (tail_length - 2) # set transparency based on t value
                ax.plot(plotting_data[n,0,t:t+2], plotting_data[n,1,t:t+2], plotting_data[n,2,t:t+2], 'b-', alpha=alpha) # plot tails

            #plotting points
            ax.scatter(plotting_data[n,0,-1], plotting_data[n,1,-1], plotting_data[n,2,-1], c="black", s=100, marker="o") # plot positions

        # Set plot labels and limits
        # create x,y,z labels + title
        ax.set_xlabel('X [kpc]')
        ax.set_ylabel('Y [kpc]')
        ax.set_zlabel('Z [kpc]')
        ax.set_title('Black Hole Trajectories')

        # set x,y,z figure limits
        ax.set_xlim( - max_range/2, max_range/2)
        ax.set_ylim( - max_range/2, max_range/2)
        ax.set_zlim( - max_range/2, max_range/2)
        
        plt.tight_layout()
        #Save current frame as png
        plt.savefig(nbuddies_path + f"/movie_dump/trajectories_{i}.png", dpi=300, bbox_inches='tight') # save fig in dir
        plt.close()

    _recompile_movie_3D(tot_nstep_eta) # Combine saved frames into video using ffmpeg

def _recompile_movie_3D(tot_nstep_eta):
    """
    Deletes movie if it exists then recreates it by compiling the pngs in movie_dump
    tot_nstep_eta: str, used to dynamically save the resulting movies with info about 
                    total time (sec), num of timesteps per batch, eta for adaptive timestep computation
    """

    if os.path.exists(nbuddies_path+f"/trajectories_{tot_nstep_eta}.mkv"): # checks if path exists
        os.remove(nbuddies_path+f"/trajectories_{tot_nstep_eta}.mkv") # if it does, remove path to old movie file
    os.system("ffmpeg -framerate 12 -start_number 0 -i "+nbuddies_path+"/movie_dump/trajectories_%01d.png -q:v 0 "+nbuddies_path+f"/trajectories_{tot_nstep_eta}.mkv") # recreate movie


def _find_last_batch_num(brute_or_tree) -> int:
    """
    finds num of last batch file saved

    Returns
    -------
    int
        num of last batch file saved
    """

    i = 0
    while os.path.exists(nbuddies_path + brute_or_tree + f"/data_batch{i}.pkl"): # while path of ith data batch exists
        i += 1 # increment i
    return i - 1 # i is number corresponding to last data batch number