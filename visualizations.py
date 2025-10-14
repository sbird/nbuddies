import os
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

nbuddies_path = os.path.dirname(os.path.realpath(__file__))

def movie_3D(tail_length : int = 10):
    """
    Loads data and makes movie of motion in 3D space with tails behind them

    Parameters
    ----------
    tail_length : int, default 10
        length of tail trailing behind points
    """
    
    #set up
    if not os.path.exist(nbuddies_path+"/movie_dump"):
        os.makedirs(nbuddies_path"/movie_dump")
    
    last_batch_num = _find_last_batch_num()
    with open(nbuddies_path + f"data_batch{last_batch_num}.pkl", 'rb') as file:
        data = pickle.load(file)
    N = len(data)
    max_range = 0
    for n in range(N)
        """revisit this line when people have made up thier minds of how data is bein saved"""
        if np.linalg.norm(data[n].position) > max_range:
            max_range = np.linalg.norm(data[n].position)
    max_range *= 1.25 # add buffer

    plotting_data = np.zeros([N, 3, tail_length])
    for n in range(N):
        for t in range(tail_length):
            plotting_data[n, :, t] = data[n].position
    
    for i in range(last_batch_num + 1):
        
        #slide data window forward
        for j in range(tail_length - 1):
            plotting_data[j] = plotting_data[j+1]
        with open(nbuddies_path + f"data_batch{i}.pkl", 'rb') as file:
            data = pickle.load(file)
        """revisit this line when people have made up thier minds of how data is bein saved"""
        for n in range(N):
            plotting_data[n, :, -1] = data[n].position

        #plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for n in range(N):
            #plotting tails
            for t in range(tail_length - 2):
                alpha = t / (tail_length - 3)
                ax.plot(plotting_data[n,0,t:t+2], plotting_data[n,1,t:t+2], plotting_data[n,2,t:t+2], 'b-', alpha=alpha)

            #plotting points
            ax.scatter(plotting_data[n,0,-1], plotting_data[n,1,-1], plotting_data[n,2,-1], c="black", s=100, marker="o")

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Black Hole Trajectories')

        ax.set_xlim( - max_range/2, max_range/2)
        ax.set_ylim( - max_range/2, max_range/2)
        ax.set_zlim( - max_range/2, max_range/2)
        
        plt.tight_layout()
        plt.savefig(nbuddied_path + f"/movie_dump/trajectories_{i}.png", dpi=300, bbox_inches='tight')

    _recompile_movie_3D()

def _recompile_movie_3D():
    """
    Deletes movie if it exists then recreates it by compiling the pngs in movie_dump
    """
    if os.path.exists(nbuddies+path+"/trajectories.mkv"):
        os.remove(nbuddies+path+"/trajectories.mkv")
    os.system(f"ffmpeg -framerate 12 -start_number 0 -i "+nbuddies_path+"/movie_dump/trajectories_%01d.png -q:v 0 "+nbuddies+path+"/trajectories.mkv")


def _find_last_batch_num():
    """
    finds num of last batch file saved

    Returns
    -------
    int
        num of last batch file saved
    """
    i = 0
    while os.path.exists(f"data_batch{i}.pkl"):
        i += 1
    return i