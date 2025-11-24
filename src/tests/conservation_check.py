"""File to run checks on (and plot the) error in energy and momentum conservation."""

from src.BlackHoles_Struct import BlackHole
import numpy as np 
from pint import UnitRegistry
import pickle
import os
import matplotlib.pyplot as plt

# set some plot style parameters
plt.rcParams['text.usetex'] = True
plt.rcParams['xtick.labelsize'] = 18  
plt.rcParams['ytick.labelsize'] = 18  

# define gravitational constant for potential energy calculation
ureg = UnitRegistry()
ureg.define('solarmass = 1.98847e30 * kilogram') 
G = 6.67430e-11 * ureg.meter**3 / (ureg.kilogram * ureg.second**2) 
GG = G.to("kiloparsec * kilometer**2 / (solarmass * second**2)") 


def compute_KE(BH_data_at_time_t):
    """
    Computes the total kinetic energy of the system at a particular time. 
    Inputs
    -------
        BH_data_at_time_t : list 
                            contains the black hole structs (length of list = no. of blackholes) at a particular time instance.

    Returns
    -------
        total_KE : pint.Quantity
                   total kinetic energy of the system in units of solarmass * km^2/sec^2
    """

    # initialize the total KE
    total_KE = 0

    # loop through all BHs in the input list, assign their velocities and masses units, and compute their KE
    for i in range(len(BH_data_at_time_t)):
        vel_BH = BH_data_at_time_t[i].velocity * ureg('km/s')
        mass_BH = BH_data_at_time_t[i].mass * ureg('solarmass')

        total_KE += 1/2 * mass_BH * np.sum(vel_BH**2)

    return total_KE

def compute_PE(BH_data_at_time_t):
    """
    Computes the total potential energy of the system at a particular time.
    Inputs
    -------
        BH_data_at_time_t : list 
                            contains the black hole structs (length of list = no. of blackholes) at a particular time instance.

    Returns
    -------
        total_KE : pint.Quantity
                   total potential energy of the system in units of solarmass * km^2/sec^2
    """
    
    # initialize the total PE
    total_PE = 0

    # loop through all BHs in the input list, assign their velocities and masses units, and compute their PE
    for i in range(len(BH_data_at_time_t)):
        for j in range(i + 1, len(BH_data_at_time_t)):
            BH_i = BH_data_at_time_t[i]
            BH_j = BH_data_at_time_t[j]    

            # obtain the position vector with units, take its magnitude and compute PE
            pos_vec = BH_i.displacement(BH_j) * ureg.kpc
            pos_mag = np.linalg.norm(pos_vec) 
            PE = - GG * BH_i.mass * BH_j.mass * (ureg.solarmass**2) / pos_mag   # in solar mass km^2 / s^2
            total_PE += PE

    return total_PE

def compute_total_energy(BH_data_at_time_t):
    """
    Computes the total energy of the system at a particular time.
    Inputs
    -------
        BH_data_at_time_t : list 
                            contains the black hole structs (length of list = no. of blackholes) at a particular time instance.

    Returns
    -------
        total_KE : pint.Quantity
                   total energy of the system in units of solarmass * km^2/sec^2
    """

    # individually compute KE and PE, and then add
    total_KE = compute_KE(BH_data_at_time_t)
    total_PE = compute_PE(BH_data_at_time_t)

    return total_KE + total_PE  

def compute_momentum(BH_data_at_time_t):
    """
    Computes the total momentum (vector) of the system at a particular time.
    Inputs
    -------
        BH_data_at_time_t : list 
                            contains the black hole structs (length of list = no. of blackholes) at a particular time instance.

    Returns
    -------
        total_momentum : pint.Quantity
                         total momentum (array of shape (3,)) of the system in units of solarmass * km/sec
    """
    
    # initialize the total_momentum
    total_momentum = np.zeros(3) * ureg('solarmass * km/s')

    # loop through all BHs in the input list, assign their velocities and masses units, and compute their momentum
    for i in range(len(BH_data_at_time_t)):
        vel_BH = BH_data_at_time_t[i].velocity * ureg('km/s')
        mass_BH = BH_data_at_time_t[i].mass * ureg('solarmass')

        total_momentum += mass_BH * vel_BH 

    return total_momentum

def get_momentum_errors(directory):
    """
    Parameters
    -----------
        directory : str
                    path to the directory containing all the data batches across timesteps.
    
    Returns 
    -------
        frac_errors_momentum : ndarray
                               array of shape (no. of batches, 3) containing the fractional errors 
                               in the momentum along each direction, for each timestep
    """

    last_batch_num = _find_last_batch_num(directory)

    tot_momentums = np.zeros((last_batch_num + 1, 3)) * ureg('solarmass * km/s')

    for i in range(last_batch_num + 1):
        with open(f'{directory}/data_batch{i}.pkl', 'rb') as file:
            data_from_batch = pickle.load(file)['data']

        tot_momentums[i] = compute_momentum(data_from_batch[-1])

        if i == 0:
            init_tot_momentum = compute_momentum(data_from_batch[0])

    frac_errors_momentum = tot_momentums/init_tot_momentum - 1

    if np.all(frac_errors_momentum < 0.05):
        print(f"momentum conservation passes with a {np.max(np.abs(frac_errors_momentum.magnitude)):.2f}% accuracy at worst")
    else:
        print("WARNING: momentum conservation fails at the 5% level along some direction")

    return frac_errors_momentum

def get_energy_errors(directory):
    """
    Parameters
    -----------
        directory : str
                    path to the directory containing all the data batches across timesteps.
    
    Returns 
    -------
        frac_errors_energy : ndarray
                               array of shape (no. of batches, ) containing the fractional errors 
                               in the energy for each timestep
    """
    
    last_batch_num = _find_last_batch_num(directory)

    tot_energies = np.zeros(last_batch_num + 1) * ureg('solarmass * km^2/s^2')

    for i in range(last_batch_num + 1):
        with open(f'{directory}/data_batch{i}.pkl', 'rb') as file:
            data_from_batch = pickle.load(file)['data']

        tot_energies[i] = compute_total_energy(data_from_batch[-1])

        if i == 0:
            init_tot_energy = compute_total_energy(data_from_batch[0])

    frac_errors_energy = tot_energies/init_tot_energy - 1
    
    if np.all(frac_errors_energy < 0.05):
        print(f"energy conservation passes with a {np.max(np.abs(frac_errors_energy.magnitude)):.2f}% accuracy at worst")
    else:
        print("WARNING: energy conservation fails at the 5% level")

    return frac_errors_energy


def plot_energy_mom_errors(frac_errors_energy, frac_errors_momentum, name_of_method : str):
    """
    Function to plot the momentum and energy fractional errors as a function of batch (i.e. time) in the simulation. 

    Parameters
    ----------
        frac_errors_energy : array_like
                             the fractional errors in energy for the simulation with shape (number of batches,)
        frac_errors_momentum : array_like
                             the fractional errors in momentum for the simulation with shape (number of batches, 3)
        name_of_method : str
                         title of the plot, possibly including specifics regarding the simulation
    
    Returns
    -------
        fig : matplotlib.figure.Figure
              plot showing the momentum and energy fractional errors as a function of batch number
    """ 
    fig = plt.figure(figsize = (10, 8))

    plt.plot(frac_errors_energy, marker = 'o', c = 'blue', label = 'Energy')
    plt.plot(np.max(np.abs(frac_errors_momentum), axis = 1), marker = 'o', c = 'r', label = 'Momentum')
    plt.axhline(0, c = 'k', ls = '--')
    plt.xlabel('Batch number', fontsize = 18)
    plt.ylabel(r'$\frac{\Delta E}{E} - 1$', fontsize = 18)
    plt.title(f'{name_of_method}', fontsize = 18)
    plt.legend(fontsize = 18)
    plt.show()
    return fig

def _find_last_batch_num(path) -> int:
    """
    finds num of last batch file saved

    Returns
    -------
    int
        num of last batch file saved
    """

    i = 0
    while os.path.exists(f"{path}/data_batch{i}.pkl"): # while path of ith data batch exists
        i += 1 # increment i
    return i - 1 # i is number corresponding to last data batch number

# example call to make the plot
if __name__ == '__main__':

    print("Energy and momentum conservation for brute force calculation - ")
    directory = 'data/mass_segregation'
    frac_errors_energy = get_energy_errors(directory = directory)
    frac_errors_momentum = get_momentum_errors(directory = directory)
    fig = plot_energy_mom_errors(frac_errors_energy, frac_errors_momentum, name_of_method='Tree_Leapfrog')