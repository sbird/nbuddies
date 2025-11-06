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

def update_params(data, tot_time, num_steps, delta_t, path, leapfrog = True, use_tree = True):
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
            result = leapfrog_integrator(data, delta_t, timestep, use_tree)
        else:
            # Euler integration
            result = euler_integrator(data, delta_t, use_tree)
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


def update_params_adaptive_timestep(data, tot_time, num_steps, eta, path, leapfrog = True, use_tree = True):
    ''' 
    Carries out the integration for each particle for one time step
    
    Inputs:
    data - list of BH objects
    eta - constant that decides the relation between timestep and acceleration, jerk, snap. Needs to be optimized.  
    tot_time - float value of the total time for evolution
    num_steps - integer value of the number of time steps to be saved in each batch
                given by batch = tot_time // (num_batches * delta_t)
    path - string of the common directory, given by the Git(?)
    leapfrog - True if want leapfrog integration, False if want simple Euler integration (default True)
    
    Output:
    None - All output files are saved as picke file
    '''

    batch_idx = 0
    count = 0 # goes from 0 to num_steps - 1, used to check when to save the data  
    data_lst = [data] # initialized with the starting data, stores the evolved data batch-wise
    running_time = 0 * ureg.sec  # time elapsed in the simulation, will end when running_time == tot_time
    # needs to be initialized with units because recalculate_acceleration now assigns units acceleration

    recalculate_dynamics(data, use_tree) # Get acceleration with current position

    while running_time.magnitude < tot_time:

        # block to decide the delta_t value for this iteration - 
        delta_t_BH = np.zeros(len(data)) * ureg.s
        for i, BH in enumerate(data):
            delta_t_BH[i] = comp_adaptive_dt(BH.acceleration, BH.jerk, BH.snap, eta)  # compute adaptive value
        delta_t = np.min(delta_t_BH)   # choose the minimum among all BHs

        print(f'time_elapsed: {running_time.magnitude/tot_time*100:.2f}% of tot_time; delta_t for this iteration: {delta_t}')
        # this statement can be useful to keep a check on how fast the simulation is proceeding for a given eta

        if (leapfrog):
            # Leapfrog Integration
            result = leapfrog_integrator(data, delta_t, running_time, use_tree)
        else:
            # Euler integration
            result = euler_integrator(data, delta_t, use_tree)
        running_time += delta_t
        count += 1
        data_lst.append(result)
        if count == num_steps:
            files = [data_lst, delta_t, tot_time, num_steps] 
            # the above way of saving means we are saving the value of timestep in the last simulation of each batch
            # will keep it like that for now so that we can access the typical values of timesteps, later can just save the eta as metadata

            save_data_pkl(files, f'data_batch{batch_idx}.pkl', path)  # saving as a pkl file right now
            batch_idx += 1
            # resets the values for the next batch
            count = 0
            data_lst = []
        
        recalculate_dynamics(data, use_tree) 
        # these need to be done before the next computation of dt (next iteration of the loop)

    # Save any remaining timesteps
    if (data_lst):
        files = [data_lst, delta_t, tot_time, num_steps]
        save_data_pkl(files, f"data_batch{batch_idx}.pkl", path)



def leapfrog_integrator(data, delta_t, timestep, use_tree):
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
        recalculate_dynamics(data, use_tree) # Get acceleartion with current position

    result = []

    # First Kick and Drift
    for BH in data:
        BH.velocity += (BH.acceleration * delta_half).magnitude # Update the velocity with half of timestep
        BH.position += ( (BH.velocity/ KM_PER_KPC) * delta_t ).magnitude # Update the position with the new velocity and with full timestep
    # need to update velocity and position without units in order to keep it compatible with the rest of the code
    # hence the .magnitude

    # Recalculation of the acceleration
    recalculate_dynamics(data, use_tree)

    # Last Kick
    for BH in data:
        BH.velocity += (BH.acceleration * delta_half).magnitude # Update the velocity with half of timestep and updated acceleration
        result.append(BH.copy())

    return result
            
def euler_integrator(data, delta_t, use_tree):
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
    recalculate_dynamics(data, use_tree)  # provided in Forces.py
    result = []
    for BH in data:  # assumes the BH objects are already loaded with initial values
        BH.position += ( (BH.velocity/ KM_PER_KPC) * delta_t ).magnitude # Euler integration (formula given above)
        BH.velocity += (BH.acceleration * delta_t).magnitude # Euler integration (formula given above)
        result.append(BH.copy())
    return result


#Function to compute adaptive timestep
def comp_adaptive_dt(acc, jerk, snap, eta):
    """
    Inputs: acc, jerk, snap, and eta 
    We will need the magnitude for acc, jerk, and snap
    Using our function from class:
    dt = eta * ((jerk/acc)**2 + (snap/acc))**(-1/2)
    Because we are dividing by acc we need to ensure that we don't divide by zero
    We can divide instead by a_mag_safe = np.maximum(a_mag, "small factor")
    Returns the adaptive timestep
    """
    #Calculates magnitudes for acc, jerk, and snap
    a_mag = np.linalg.norm(acc)
    j_mag = np.linalg.norm(jerk)
    s_mag = np.linalg.norm(snap)

    dt = eta / np.sqrt((j_mag / a_mag)**2 + (s_mag / a_mag)) #computes dt 

    # print("a_mag = ", a_mag)
    # print("j_mag = ", j_mag)
    # print("s_mag = ", s_mag)  # these will have proper units attached with them
    # print('dt =', dt) # this will have the correct units (seconds) as well

    return dt


def simulation(initial_file, output_folder, tot_time, nsteps, delta_t = None, adaptive_dt = False, eta = None, use_tree = True):
    """
    Wrapper Function for the simulation of time evolve N-body Problem
    
    Inputs:
    initial_file : Path name to the file contained the initial condition file
    output_folder : Path to folder for the save time steps
    tot_time : total amount of time of the simulation
    delta_t : size of the timestep
    nsteps : number of steps for each saving of the batch
    adaptive_dt: whether to use the adaptive timestep formula using eta
    eta: the constant for the adpative timestep formula. Cannot be none if adpative_dt is True. 
    use_tree : whether to use the BHT code for force calculation (default True)
    
    Outputs:
        
    """

    # load initial condition
    data, inital = load_data_pkl(initial_file) # should be a list of BH objects

    # Run Simulation
    if adaptive_dt:
        if eta is None:
            raise ValueError("Adaptive timestepping (adaptive_dt = True) requires a value of eta to be given.")
        else:
            update_params_adaptive_timestep(data, tot_time, nsteps, eta, output_folder, use_tree)
    else:
        update_params(data, tot_time, nsteps, delta_t, output_folder, use_tree)


print('Yay! The evolution2.py file is being used!')
# print('\nNeed to call the simulation function properly to ensure it works though :)')

# Example usage by calling the simulation function using arbitrary parameters and names
# simulation1 = simulation('initial_conditions.pkl', './', 100, 0.01, 10)