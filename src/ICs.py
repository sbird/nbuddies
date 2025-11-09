import numpy as np
import pickle as pkl
from .BlackHoles_Struct import BlackHole
import os

path_to_save_pkl_file = "./"
GG = 4.301E-6 # Newton's constant [km^2*kpc / solar mass*s^2]

# def generate_initial_conditions(n_blackholes, box_size, mass_range = (3, 10**11), vel_scale_kms = 100): # a function to create n_blackholes with masses, positions, and velocities, box_size in kpc (side length of the cubic box)
#     '''
#     Preliminary Way Initial Conditions Were Obtained
#     '''
#     masses = np.random.uniform(mass_range[0], mass_range[1], n_blackholes) # Makes n_blackholes random masses, uniformly between the given min and max
#     positions = np.random.uniform(-box_size/2, box_size/2 , (n_blackholes, 3)) # (a, b, (n,m)) makes an array of shape (n,m) with random numbers uniformly between a and b. 
#     velocities = np.zeros((n_blackholes, 3)) # Prepares an array to hold all velocities (km/s), initially all zeros. Make three random numbers from a bell curve with mean 0 and spread sigma 

#     for i in range(n_blackholes):
#         sigma = vel_scale_kms / (np.sqrt(masses[i] / np.mean(masses)))  # scale velocity dispersion with mass
#         velocities[i] = np.random.normal(0, sigma, 3)  # 3D velocity components # mean velocity = 0 (random normal distribution)

#     blackholes = []
#     for i in range(n_blackholes):
#         bh = BlackHole(masses[i], positions[i], velocities[i])
#         blackholes.append(bh)

#     pkl.dump(blackholes, open(f"{path_to_save_pkl_file}/test1.pkl", "wb")) # save the blackholes list to a pickle file    
#     return blackholes, masses, positions, velocities    
'''
Start of active code 
'''
# The functions below are based on the paper Aarseth et al. 1974
# https://articles.adsabs.harvard.edu/pdf/1974A%26A....37..183A

def generate_mass(n: int, initial_mass: float, ratio: float) -> list[float]:
    """
    Generate mass [solar mass] of n black holes such that there are two types of masses
    Input:
        n, number of black holes
        initial_mass, mass [solar mass] of first type of black holes
        ratio, mass ratio between two types of black holes
    Output:
        list[mass], mass [solar mass] of each black hole
    """
    mass_1 = initial_mass
    mass_2 = initial_mass * 10
    n0 = n - np.round(ratio*n)
    assert ratio <= 0.1, "Ratio must be smaller than 0.1"
    x = (n * mass_1) / (n0 * mass_1 + (n-n0) * mass_2) # Normalization factor to make sure total mass is similar to n*initial_mass

    mass = np.zeros(n)
    for i in range (n):
        if i < n0:
            mass[i] = mass_1 * x
        else:
            mass[i] = mass_2 * x
    return mass 

def generate_radius(a: float) -> float:
    """
    Generate the radius [kpc] of a black hole using a random number from a uniform probability distribution
    Input:
        a, scale black hole radius to kpc
    Output:
        radius of black hole
    """
    x = np.random.uniform()
    return a*(x**(-2/3) - 1)**(-1/2) # Derived from Equation (A2)

def generate_random_vector_of_magnitude(magnitude : float) -> list[float]:
    """
    Generate a random vector from istropic probability distribution
    
    Parameters
    ----------
    magnitude
        magnitude of random vector
    
    Returns
    -------
    list[float]
        a random vector of desired magnitude
    """
    x_1 = np.random.uniform()
    x_2 = np.random.uniform()
    z = (1 - 2*x_1)*magnitude # Equation (A3)
    x = np.sqrt(magnitude**2 - z**2) * np.cos(2*np.pi*x_2)
    y = np.sqrt(magnitude**2 - z**2) * np.sin(2*np.pi*x_2)
    return [x, y, z]

def find_q() -> float:
    """
    Find velocity modulus q using von Neumann's rejection technique
    Output:
        x_1, random number sampled from a uniform probability distribution
    """
    x_1 = np.random.uniform()
    x_2 = np.random.uniform()
    if 0.1 * x_2 < g(x_1):
        return x_1
    else:
        return find_q()

def g(q: float) -> float:
    """
    Generate the probability distribution that is proportional to velocity modulus q derived from Equation (4)
    Input:
        q, velocity modulus 
    Output:
        probability distribution of q
    """
    return q**2*(1 - q**2)**(7/2) # Equation (A5)

def calculate_escape_velocity(radius: float, M: float, a: float) -> float:
    """
    Calculate escape velocity [km/s]
    Input:
        radius, radius [kpc] of black hole
        M, total mass [solar mass] of black holes
        a, scale black hole radius to kpc
    Output:
        escape velocity
    """
    return np.sqrt(2*GG*M) * (a**2 + radius**2)**(-1/4) # Derived from Equation (A4)

def generate_plummer_initial_conditions(n_blackholes: int, initial_mass: float, scale: float, ratio = 0.0) -> tuple[list[BlackHole], float]: 
    """
    A function to create n_blackholes with positions, velocities, and equal masses
    by generating initial coniditions for N-body black hole simulation using the Plummer model.
    Inputs:
        n_blackholes, number of black holes to generate
        mass, mass [solar mass] of each black hole
        scale, radius [kpc] scale of black holes
        ratio, how many of the black holes are 10 times more massive than the others
    Outputs:
        blackholes, black hole objects generated by this function
        mass, mass [solar mass] of each BlackHole object
        positions, position [kpc] vector of each BlackHole object
        velocities, velocity [km/s] vector of each BlackHole object
    """    
    blackholes = np.empty(n_blackholes, BlackHole) # Prepares an array to hold all blackholes  
    masses = generate_mass(n_blackholes, initial_mass, ratio)

    # an example of positions: [[x1,y1,z1],[x2,y2,z2],...,[xn,yn,zn]], 
    # where xi,yi,zi are the coordinates of the i-th black hole  

    for i in range(n_blackholes):
        r = generate_radius(scale)

        v_esc = calculate_escape_velocity(r, np.sum(masses), scale)
        q = find_q()
        v = q*v_esc
        
        blackholes[i] = BlackHole(
            masses[i],
            generate_random_vector_of_magnitude(r),
            generate_random_vector_of_magnitude(v)
        )
    blackholes_dict = {'data': blackholes}
    
    return blackholes_dict, masses   

def generate_binary_ICs( N_BH: int, custom_vals: dict = 
                        {   'N': 2,
                            'mass': np.array([1.0e7, 1.0e7]), 
                            'position': np.array([[1., 0., 0.], [-1., 0., 0.]]), 
                            'velocity': np.array([[0. ,3.2791 ,0.], [0. ,-3.2791 ,0.]])} ) -> list[float]:
    # outlines the structure expected from the ICs team
    # we demand that there be a custom_vals option as defined here, so that we can test out 
    # simple cases corresponding to specific (instead of random) initial positions and velocities
    """
    inputs:
    N_BH: total number of black holes in the simulation
    custom_vals: contains the user-specified values for masses, positions and velocities. {'mass' : np array shape (N_BH, ), 'position' : np array shape (N_BH, 3) ,'velocity' : np array shape (N_BH, 3)}

    outputs:
    BH_data_ic: a list of BH objects initialized with the initial values of mass, positions, and velocities
    """

    # this is the 'black hole data', and will eventually store the BH objects after initialization
    # BH_data_ic = []

    # if they are assigned as separate np arrays, 
    init_BH_masses = custom_vals['mass']    
    init_BH_positions = custom_vals['position']
    init_BH_velocities = custom_vals['velocity']

    # the above choice depends on how the ics team have implemented it

    ## Load the custom_vals into Class objects
    list_of_BH = []
    for i in range( N_BH ):
        BH = BlackHole( init_BH_masses[i], init_BH_positions[i], init_BH_velocities[i] )
        list_of_BH.append(BH)

    blackholes_dict = {'data': list_of_BH}

    return blackholes_dict, init_BH_masses

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(43)
    
    # Generate initial conditions for 20 black holes
    n = 20              # number of black holes
    mass = 1e6          # solar masses per BH
    m1_ratio = 0.05      # mass ratio between two types of black holes
    scale = 1           # scale (a value)
    blackholes, masses = generate_plummer_initial_conditions(n, mass, scale, m1_ratio)

    # Provides the mass, position, and velocity for each black holes
    vmin = np.inf
    vmax = -np.inf
    pmin = np.inf
    pmax = -np.inf

    # blackholes = blackholes['data']
    for i in range(n):
        print(f"\nBlack hole {i+1}:")
        print(f"  Mass: {blackholes[i].mass:.1f} M_sun")
        print(f"  Position: {blackholes[i].position} kpc")
        print(f"  Velocity: {blackholes[i].velocity} km/s")

        vmag = np.linalg.norm(blackholes[i].velocity)
        pmag = np.linalg.norm(blackholes[i].position)
        vmin = min(vmin, vmag)
        vmax = max(vmax, vmag)
        pmin = min(pmin, pmag)
        pmax = max(pmax, pmag)

    # # Prints the number of black holes generated and the range of positions and velocities generated
    print(f"\nVelocity magnitude range: [{vmin:.2f}, {vmax:.2f}] km/s")
    print(f"Position magnitude range: [{pmin:.2f}, {pmax:.2f}] kpc")
    print(f"Total mass: {np.sum(masses):.1f} M_sun")