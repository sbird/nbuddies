from Forces import *
import os
import pickle
import numpy as np

KM_PER_KPC = 3.0856776e16 # number of km in kpc for using velocity to update position

def load_data_pkl(filename, path = None):
    ''' 
    Load the input position or velocity from a pickle file
    
    Inputs: 
    filename - string of the filename, to be given by the IC team
    path - string of the common directory, given by the Git(?)

    Output:
    Array of the loaded data having shape (N,3),
    where N is number of particles, 3 is the coordinates
    or
    list of Blackhole class objects having shape (batch_size, N) along with the metadata
    '''
    if (path is None):
        file_path = filename
    else:
        file_path = os.path.join(path, filename)
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    '''
    checking if the loaded file is the dictionary of the data of blackholes with metadata, 
    or a simple pickle file with information given by IC team
    '''
    if isinstance(data, dict):
        if "data" in data:
            print("Loaded pickle file with metadata.")
            return data["data"], data          #returns both the data and the metadata
        else:
            print("Loaded pickle file without metadata.")
            return data                        #returns the data only - useful for IC team
    else: 
        raise ValueError(f"Expected a .pkl file, got: {os.path.basename(file_path)}")

def save_data_pkl(files, filename, path):
    ''' 
    Save the updated position or velocity arrays as pickle files
    
    Inputs:
    filename - string of the filename, chosen by the timesteps team
    path - string of the common directory, given by the Git(?)
    data - list of Blackhole class objects having shape (batch_size, N) 

    Output:
    None
    '''
    # check if the folder exit
    os.makedirs(path) if not os.path.exists(path) else None 
    # join folder with filename
    file_path = os.path.join(path, filename)

    data, delta_t, tot_time, num_steps = files # unpack the input tuple
    if data is None:
        print("No time-evolved data to save")   #if used by the IC team
    with open(file_path, 'wb') as f:
        pickle.dump({
            "time units" : "s",
            "distance units" : "kpc",
            "velocity units" : "km/s",
            "number of particles" : len(data[0]), # number of particles in each batch
            "total time" : tot_time,
            "number of steps per batch" : num_steps,
            "delta_t" : delta_t,
            "data" : data
        }, f)

def update_params(data, tot_time, num_steps, delta_t, path, leapfrog = True):
    ''' 
    Carries out the integration for each particle for one time step
    
    Inputs:
    data - list of BH objects
    delta_t - float value of the time step for evolution
    tot_time - float value of the total time for evolution
    num_steps - integer value of the number of time steps to be saved in each batch
                given by batch = tot_time // (num_steps * delta_t)
    path - string of the common directory, given by the Git(?)
    leapfrog - True if want leapfrog integration, False if want simple Euler integration (default True)
    
    Output:
    None - All output files are saved as picke file
    '''

    batch_idx = 0
    count = 0 # goes from 0 to num_steps - 1, used to check when to save the data  
    data_lst = [data] # initialized with the starting data, stores the evolved data batch-wise
    for i, timestep in enumerate(np.arange(0, tot_time, delta_t)): # for each time step, carry out the evolution for all BHs
        if (leapfrog):
            # Leapfrog Integration
            result = leapfrog_integrator(data, delta_t, timestep)
        else:
            # Euler integration
            result = euler_integrator(data, delta_t)
        count += 1
        data_lst.append(result)
        if count == num_steps:
            files = [data_lst, delta_t, tot_time, num_steps]
            save_data_pkl(files, f'data_batch{batch_idx}.pkl', path)  # saving as a pkl file right now
            batch_idx += 1
            # resets the values for the next batch
            count = 0
            data_lst = []
    
    # Save any remaining timesteps
    if (data_lst):
        files = [data_lst, delta_t, tot_time, num_steps]
        save_data_pkl(files, f"data_batch{batch_idx}.pkl", path)

def leapfrog_integrator(data, delta_t, timestep):
    """
    Updating position and velocity of BH objects with conserving phase space volume (symplectic integrator).
    
    1. First Kick: Update velocity by half step using current acceleration
    2. Drift: Update position by full step using updated velocity
    3. Recalculate acceleration with new positions
    4. Second Kick: Update velocity by another half step using new acceleration
    
    Inputs:
    data - list of BH objects
    delta_t - The timestep for each round of update
    time_step - current time step index (to check if acceleration needs recalculation)
    
    Output:
    result - list of Blackhole object
    
    """
    delta_half = delta_t / 2
    if timestep == 0:
        recalculate_accelerations(data) # Get acceleartion with current position

    result = []

    # First Kick and Drift
    for BH in data:
        BH.velocity += BH.acceleration * delta_half # Update the velocity with half of timestep
        BH.position += (BH.velocity/ KM_PER_KPC) * delta_t # Update the position with the new velocity and with full timestep

    # Recalculation of the acceleration
    recalculate_accelerations(data)

    # Last Kick
    for BH in data:
        BH.velocity += BH.acceleration * delta_half # Update the velocity with half of timestep and updated acceleration
        result.append(BH.copy())

    return result
            
def euler_integrator(data, delta_t):
    """
    Euler integration of the position and velocity according to
    r(t+Delta t) = r(t) + v(t) * Delta t
    v(t+Delta t) = v(t) + a(t) * Delta t

    Inputs:
    data - list of BH objects
    delta_t - The timestep for each round of update
    
    Output:
    result - list of Blackhole object
    
    """
    recalculate_accelerations(data)  # provided in Forces.py
    result = []
    for BH in data:  # assumes the BH objects are already loaded with initial values
        BH.position += (BH.velocity/ KM_PER_KPC) * delta_t # Euler integration (formula given above)
        BH.velocity += BH.acceleration * delta_t # Euler integration (formula given above)
        result.append(BH.copy())
    return result

def simulation(initial_file, output_folder, tot_time, delta_t, nsteps):
    """
    Wrapper Function for the simulation of time evolve N-body Problem
    
    Inputs:
    initial_file : Path name to the file contained the initial condition file
    output_folder : Path to folder for the save time steps
    tot_time : total amount of time of the simulation
    delta_t : size of the timestep
    nsteps : number of steps for each saving of the batch
    
    Outputs:
        
    """
    # load initial condition
    data, inital = load_data_pkl(initial_file) # should be a list of BH objects

    # Run Simulation
    update_params(data, tot_time, nsteps, delta_t, output_folder)

print('Yay! The evolution2.py file is being used!')
# print('\nNeed to call the simulation function properly to ensure it works though :)')

# Example usage by calling the simulation function using arbitrary parameters and names
# simulation1 = simulation('initial_conditions.pkl', './', 100, 0.01, 10)