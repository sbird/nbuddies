from Forces import recalculate_accelerations
import os
import pickle
import numpy as np

def load_data_pkl(filename, path = None):
    ''' 
    Load the input position or velocity from a pickle file
    
    Inputs: 
    filename - string of the filename, to be given by the IC team
    path - string of the common directory, given by the Git(?)

    Output:
    Array of the loaded data having shape (N,3),
    where N is number of particles,
    3 is the coordinates
    '''
    if (path == None):
        file_path = filename
    else:
        file_path = os.path.join(path, filename)
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def save_data_pkl(data, filename, path):
    ''' 
    Save the updated position or velocity arrays as pickle files
    
    Inputs:
    filename - string of the filename, chosen by the timesteps team
    path - string of the common directory, given by the Git(?)
    data - numpy arrays/class objects of shape (B, 3) containing position or velocity.

    Output:
    None
    '''
    file_path = os.path.join(path, filename)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def load_data_np(filename, path):
    ''' 
    Load the input position or velocity arrays as numpy files
    
    Inputs: 
    filename - string of the filename, to be given by the IC team
    path - string of the common directory, given by the Git(?)

    Output:
    Array of the loaded data having shape (N,3),
    where N is number of particles,
    3 is the coordinates
    '''
    return np.load(path+filename)


def save_data_np(data, filename, path):
    ''' 
    Save the updated position or velocity arrays as numpy files
    
    Inputs:
    filename - string of the filename, chosen by the timesteps team
    path - string of the common directory, given by the Git(?)
    data - numpy arrays/class objects of shape (B, 3) containing position or velocity.

    Output:
    None
    '''
    return np.save(data, path+filename)

def update_params(data, tot_time, num_steps, delta_t, path):
    ''' 
    Euler integration of the position and velocity according to
    r(t+Delta t) = r(t) + v(t) * Delta t
    v(t+Delta t) = v(t) + a(t) * Delta t
    Carries out the integration for each particle for one time step
    
    Inputs:
    data - list of BH objects
    delta_t - float value of the time step for evolution
    tot_time - float value of the total time for evolution
    num_steps - integer value of the number of time steps to be saved in each batch
    path - string of the common directory, given by the Git(?)
    
    Output:
    None
    '''

    batch = tot_time // num_steps # number of batches
    count = 0 # goes from 0 to num_steps - 1, used to check when to save the data
    data_lst = [data] # initialized with the starting data, stores the evolved data batch-wise
    KM_PER_KPC = 3.0856776e16 # number of km in kpc for using velocity to update position
    for time_step in range(0, tot_time, delta_t): # for each time step, carry out the evolution for all BHs
        recalculate_accelerations(data)  # provided in Forces.py by the Forces team
        for BH in data:  # assumes the BH objects are already loaded with initial values
            BH.position += (BH.velocity / KM_PER_KPC) * delta_t # Euler integration (formula given above)
            BH.velocity += BH.acceleration * delta_t # Euler integration (formula given above)
        count += 1
        data_lst.append(data)
        if count == batch:
            save_data_pkl(data_lst, f'data_batch{(time_step+1)//num_steps}.pkl', path)  # saving as a pkl file right now
            # resets the values for the next batch
            count = 0
            data_lst = []

def simulation(initial_file, output_folder, period, delta_t, nsteps):
    """
    Wrapper Function for the simulation of time evolve N-body Problem
    
    Inputs:
    initial_file : Pathname to the file contained the initial condition file
    output_folder : Path to folder for the save time steps
    period : total amount of time of the simulation
    delta_t : timesteps
    nsteps : number of steps for each saving
    
    Outputs:
    None
    """
    # load initial condition
    inital = load_data_pkl(initial_file) # should be a list of BH objects

    # Run Simulation
    update_params(inital, period, nsteps, delta_t, output_folder)

    
            

# def update_params(data, tot_time, num_steps, delta_t):
#     ''' 
#     Euler integration of the position and velocity according to
#     r(t+Delta t) = r(t) + v(t) * Delta t
#     v(t+Delta t) = v(t) + a(t) * Delta t
#     Carries out the integration for each particle for one time step
    
#     Inputs:
#     data - list of BH objects
#     delta_t - float value of the time step for evolution
#     tot_time - float value of the total time for evolution
#     num_steps - integer value of the number of time steps to be saved in each batch
    
#     Output:
#     None
#     '''
#     batch = tot_time // num_steps # number of batches
#     count = 0 # goes from 0 to num_steps - 1
#     pos_array = np.zeros(num_steps, len(data), 3) # array to store the position data of each batch
#     vel_array = np.zeros(num_steps, len(data), 3) # array to store the velocity data of each batch
    
#     for i, BH in enumerate(data):  # assumes the BH objects are already loaded with initial values
#         pos_array[0][i] = BH.position
#         vel_array[0][i] = BH.velocity
#         for time_step in range(tot_time): 
#             recalculate_acceleration_due_to_gravity(data)
#             BH.position += BH.velocity * delta_t
#             BH.velocity += BH.acceleration * delta_t
#             count += 1
#             pos_array[count][i] = BH.position
#             vel_array[count][i] = BH.velocity
#             if count == batch:
#                 save_data_pkl(pos_array, f'pos_batch{(time_step+1)//num_steps}_body{i+1}.pkl', path)
#                 save_data_pkl(vel_array, f'vel_batch{(time_step+1)//num_steps}_body{i+1}.pkl', path)
#                 count = 0
#                 pos_array[0][i] = BH.position
#                 vel_array[0][i] = BH.velocity
            
#     # return data

print('Yay! The evolution2.py file is being used!')