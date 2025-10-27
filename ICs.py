import numpy as np
import pickle as pkl
from BlackHoles_Struct import BlackHole

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

def generate_position(radius: float) -> list[float]:
    """
    Generate the position [kpc] vector with components (x, y, z) from uniform probability distribution
    Input:
        radius, radius [kpc] of black hole
    Output:
        position vector [x, y, z]
    """
    x_1 = np.random.uniform()
    x_2 = np.random.uniform()
    z = (1 - 2*x_1)*radius # Equation (A3)
    x = np.sqrt(radius**2 - z**2) * np.cos(2*np.pi*x_2)
    y = np.sqrt(radius**2 - z**2) * np.sin(2*np.pi*x_2)
    return [x, y, z]

def generate_velocity(velocity: float) -> list[float]:
    """
    Generate the velocity [km/s] vector with components (u, v, w) from uniform probability distribution
    Input:
        velocity, escape velocity [km/s] from calling calculate_velocity()
    Output:
        velocity vector [u, v, w]
    """
    x_1 = np.random.uniform()
    x_2 = np.random.uniform()
    w = (1 - 2*x_1)*velocity # Equation (A6)
    u = np.sqrt(velocity**2 - w**2) * np.cos(2 *np.pi*x_2)
    v = np.sqrt(velocity**2 - w**2) * np.sin(2 *np.pi*x_2)
    return [u, v, w]

def calculate_velocity(v_esc: float, q: float) -> float:
    """
    Calculate velocity [km/s] using velocity modulus and escape velocity [km/s]
    Inputs:
        v_esc, escape velocity from calculate_escape_velocity(), and q, velocity modulus from find_q()
    Output, 
        velocity
    """
    return q * v_esc

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
    Generate the probability distribution that is proportionalto velocity modulus q derived from Equation (4)
    Input:
        q, velocity modulus 
    Output:
        probability distribution of q
    """
    return q**2*(1 - q**2)**(7/2) # Equation (A5)

def calculate_escape_velocity(radius: float, N: int, m: float, a: float) -> float:
    """
    Calculate escape velocity [km/s]
    Input:
        radius, radius [kpc] of black hole
        N, number of black holes
        m, mass [solar mass] of black holes such that total mass is mN
        a, scale black hole radius to kpc
    Output:
        escape velocity
    """
    return np.sqrt(2*GG*N*m) * (a**2 + radius**2)**(-1/4) # Derived from Equation (A4)

def generate_plummer_initial_conditions(n_blackholes: int, mass: float, scale: float) -> tuple[BlackHole, float, list[float], list[float]]: 
    """
    A function to create n_blackholes with positions, velocities, and equal masses
    by generating initial coniditions for N-body black hole simulation using the Plummer model.
    Inputs:
        n_blackholes, number of black holes to generate
        mass, mass [solar mass] of each black hole
        scale, radius [kpc] scale of black holes
    Outputs:
        blackholes, black hole objects generated by this function
        mass, mass [solar mass] of each BlackHole object
        positions, position [kpc] vector of each BlackHole object
        velocities, velocity [km/s] vector of each BlackHole object
    """    
    positions = np.zeros((n_blackholes, 3)) # Prepares an array to hold all positions, initially all zeros. 
                                            # These are the (x, y, z) for that black hole.
    velocities = np.zeros((n_blackholes, 3)) # Prepares an array to hold all velocities, initially all zeros. 
                                             # These are the (vx, vy, vz) for that black hole.    
   
    # an example of positions: [[x1,y1,z1],[x2,y2,z2],...,[xn,yn,zn]], 
    # where xi,yi,zi are the coordinates of the i-th black hole  
    for i in range(n_blackholes):
        r = generate_radius(scale)
        positions[i] = generate_position(r)

        v_esc = calculate_escape_velocity(r, i, mass, scale)
        q = find_q()
        v = calculate_velocity(v_esc, q)
        velocities[i] = generate_velocity(v)

    # create black hole instances
    blackholes = []
    for i in range(n_blackholes):
        bh = BlackHole(mass, positions[i], velocities[i])
        blackholes.append(bh)    

    pkl.dump(blackholes, open(f"{path_to_save_pkl_file}/test_plummer1.pkl", "wb")) # save the blackholes list to a pickle file    
    return blackholes, mass, positions, velocities    

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(43)
    
    # Generate initial conditions for 20 black holes
    n = 20 # number of black holes
    mass = 1e6        # solar masses per BH
    scale = 1       #scale (a value) 
    blackholes, masses, positions, velocities = generate_plummer_initial_conditions(n, mass, scale)
    pkl.dump(blackholes, open(f"{path_to_save_pkl_file}/test1.pkl", "wb"))
    
    # Prints the number of black holes generated and the range of positions and velocities generated
    print(f"Generated {n} black holes")
    print(f"Position range: [{positions.min():.2f}, {positions.max():.2f}] kpc")
    print(f"Velocity range: [{velocities.min():.2f}, {velocities.max():.2f}] km/s")
    
    # Provides the mass, position, and velocity for each black holes
    for i in range(n):
        print(f"\nBlack hole {i+1}:")
        print(f"  Mass: {blackholes[i].mass:.1f} M_sun")
        print(f"  Position: {blackholes[i].position} kpc")
        print(f"  Velocity: {blackholes[i].velocity} km/s")