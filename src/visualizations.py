import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np


nbuddies_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

def movie_3D(sim_name : str, tail_length: int = 10, tot_nstep_eta = None):
    """
    Loads data and makes movie of motion in 3D space with tails behind them

    Parameters
    ----------
    sim_name : str
        The name of the simulation run
    tail_length : int, default 10
        length of tail trailing behind points
    tot_nstep_eta: str, used to dynamically save the resulting movies with info about 
                    total time (sec), num of timesteps per batch, eta for adaptive timestep computation
    """
    
    #set up
    # Creating output folder for animation frames if it doesn't exist
    if not os.path.exists(nbuddies_path+"/movie_dump/"+sim_name): # check if dir exists
        os.makedirs(nbuddies_path+"/movie_dump/"+sim_name) # if not, create dir path
    
    #getting info from sim end
    last_batch_num = _find_last_batch_num(sim_name) # find number corresponding to last data batch number
    
    # Load last batch and initialize data structures
    with open(nbuddies_path + '/data/' + sim_name + f"/data_batch{last_batch_num}.pkl", 'rb') as file: # open dir of batches
        data = pickle.load(file)['data'][0] # load pk data files with last batch number
    N = len(data) # length of data files

    #Calculate the maximum plot range based on particle positions
    max_range = 0
    max_mass = 0.0
    for n in range(N):
        max_mass = max(max_mass, data[n].mass)
        if np.linalg.norm(data[n].position) > max_range: # if Euclidian distance is greater than max_range
            max_range = np.linalg.norm(data[n].position) # set max_range to Euclidian distance
    max_range *= 2 # add buffer by increasing max_range by 25%

    #getting info from sim start
    with open(nbuddies_path + '/data/'+ sim_name + "/data_batch0.pkl", 'rb') as file:
        init_data = pickle.load(file)['data'][0]
    #Create 3D array to store tail positions
    plotting_data = np.zeros([N, 3, tail_length]) # instantiate array of zeros with dimensions N x 3 x tail_length
    min_mass = np.inf
    for n in range(N):
        min_mass = min(min_mass, init_data[n].mass)
        for t in range(tail_length):
            plotting_data[n, :, t] = init_data[n].position # append positions to plotting_data
    
    #init mass array
    masses = np.zeros(N)

    #set up cmap
    viridis_dark = colors.LinearSegmentedColormap.from_list('viridis_dark', 
                                                plt.cm.viridis(np.linspace(0, 0.7, 256))).reversed()

    #generating movie frames
    #Loop through batch to make frames
    for i in range(last_batch_num + 1):
        
        #slide data window forward
        for j in range(tail_length - 1):
            plotting_data[:,:,j] = plotting_data[:,:,j+1] # move data window forward by 1
        #Load current frame data
        with open(nbuddies_path + '/data/' + sim_name + f"/data_batch{i}.pkl", 'rb') as file: # open dir to pk files
            file = pickle.load(file)
            data = file['data'][0] # load pk files
            time = file['time'][0]

        #Update tail with current particle positions and masses
        for n in range(N):
            plotting_data[n, :, -1] = data[n].position
            masses[n] = data[n].mass

        #plot
        fig = plt.figure() # instantiate figure
        ax = fig.add_subplot(111, projection='3d') # create 3D subplot

        for n in range(N):
            #plotting tails
            for t in range(tail_length - 1):
                alpha = t / (tail_length - 2) # set transparency based on t value
                ax.plot(plotting_data[n,0,t:t+2], plotting_data[n,1,t:t+2], plotting_data[n,2,t:t+2], 'b-', alpha=alpha) # plot tails

        #plotting points
        points = ax.scatter(plotting_data[:,0,-1], plotting_data[:,1,-1], plotting_data[:,2,-1], c=masses, s=100/N, marker="o", vmin=min_mass, vmax=max_mass, cmap = viridis_dark, alpha=1) # plot positions

        #assign colorbar
        cbar = plt.colorbar(points, ax=ax, pad=0.1)
        cbar.set_label(r"Mass $M_{\odot}$")

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

        ax.set_title(f"t={time.to('Myr'):.3}")
        
        plt.tight_layout()
        #Save current frame as png
        plt.savefig(nbuddies_path + "/movie_dump/"+sim_name+f"/trajectories_{i}.png", dpi=300, bbox_inches='tight') # save fig in dir
        plt.close()

    _recompile_movie_3D(sim_name, tot_nstep_eta) # Combine saved frames into video using ffmpeg

def _recompile_movie_3D(sim_name, tot_nstep_eta):
    """
    Deletes movie if it exists then recreates it by compiling the pngs in movie_dump

    Parameters
    ----------
    sim_nam : str
        name of simulation
    tot_nstep_eta: str
        used to dynamically save the resulting movies with info about total time (sec), num of timesteps per batch, eta for adaptive timestep computation
    """
    if not os.path.exists(nbuddies_path+"/visuals/"+sim_name):
        os.makedirs(nbuddies_path+"/visuals/"+sim_name)

    if os.path.exists(nbuddies_path + "/visuals/" + sim_name + f"/trajectories_{tot_nstep_eta}.mkv"): # checks if path exists
        os.remove(nbuddies_path + "/visuals/" + sim_name + f"/trajectories_{tot_nstep_eta}.mkv") # if it does, remove path to old movie file
    os.system("ffmpeg -framerate 12 -start_number 0 -i " + nbuddies_path + "/movie_dump/"+ sim_name +"/trajectories_%01d.png -q:v 0 " + nbuddies_path + "/visuals/" + sim_name + f"/trajectories_{tot_nstep_eta}.mkv") # recreate movie


def _find_last_batch_num(sim_name) -> int:
    """
    finds num of last batch file saved

    Parameters
    ----------
    sim_name : str
        The name of the simulation
    Returns
    -------
    int
        num of last batch file saved
    """

    i = 0
    while os.path.exists(nbuddies_path + "/data/" + sim_name + f"/data_batch{i}.pkl"): # while path of ith data batch exists
        i += 1 # increment i
    return i - 1 # i is number corresponding to last data batch number


def radial_position_plot(sim_name):
    last_batch_num = _find_last_batch_num(sim_name)

    #getting info from sim start
    with open(nbuddies_path + '/data/'+ sim_name + "/data_batch0.pkl", 'rb') as file:
        init_data = pickle.load(file)['data']
    
    n_batch = len(init_data)
    N = len(init_data[0])

    r_points = np.zeros([N, last_batch_num*n_batch])
    t_points = np.zeros(last_batch_num*n_batch)

    masses = np.zeros(N)

    for n in range(N):
        masses[n] = init_data[0][n].mass

    for i in range(last_batch_num):
        with open(nbuddies_path + '/data/'+ sim_name + "/data_batch0.pkl", 'rb') as file:
            file = pickle.load(file)
        for j in range(n_batch):
            k = i*n_batch + j
            for n in range(N):
                r_points[n,k] = np.linalg.norm(file["data"][j][n].position)
            t_points[k] = file["time"][j].to('Myr').magnitude
    
    #set up cmap
    viridis_dark = colors.LinearSegmentedColormap.from_list('viridis_dark', plt.cm.viridis(np.linspace(0, 0.7, 256))).reversed()

    norm = colors.Normalize(vmin=np.min(masses), vmax=np.max(masses))

    fig = plt.figure()
    ax = fig.add_subplot()

    line_colors = viridis_dark(norm(masses))

    for n in range(N):
        ax.plot(t_points, r_points[n], color=line_colors[n])

    ax.set_xlabel("t (Myr)")
    ax.set_ylabel("r (kpc)")
    ax.set_title("Radial Position over Time")

    sm = plt.cm.ScalarMappable(cmap=viridis_dark, norm=norm)
    sm.set_array([])  # This line is needed for colorbar to work
    cbar = plt.colorbar(sm, ax=ax, pad=0.1)
    cbar.set_label(r"Mass $M_{\odot}$")

    plt.tight_layout()
    plt.savefig(nbuddies_path + "/visuals/" + sim_name + "/radial_positions.png")

    