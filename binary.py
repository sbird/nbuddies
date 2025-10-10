import numpy as np 
from pint import UnitRegistry
from BlackHoles_Struct import BlackHole
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

### PLAN - 

### 1) assign initial m, x and v to the two black holes (flag for random (from ICs team) vs custom) 

### 2) call the evolve_BH function from timesteps team (set inputs such as total timesteps)

### 3) check against analytical solution (need to write a plotting_function)

# Step 1 

# this is the function expected from ics team

def generate_ICs(N_BH, custom_vals = None):
    # outlines the structure expected from the ICs team
    # we demand that there be a custom_vals option as defined here, so that we can test out 
    # simple cases corresponding to specific (instead of random) initial positions and velocities
    """
    inputs:
    N_BH: total number of black holes in the simulation
    custom_vals: contains the user-specified values for masses, positions and velocities. 
                 should be a dictionary like {'mass' : np array shape (N_BH, ), 'postion' : np array shape (N_BH, 3) ,'velocity' : np array shape (N_BH, 3)} 
    
    outputs:
    BH_data_ic: a list of BH objects initialized with the initial values of mass, positions, and velocities
    """

    # this is the 'black hole data', and will eventually store the BH objects after initialization
    # BH_data_ic = []

    if custom_vals != None:
        # if every physical quantity is in the same dict, 
        init_BH_values = custom_vals    

        # if they are assigned as separate np arrays, 
        init_BH_masses = custom_vals['mass']    
        init_BH_positions = custom_vals['position']
        init_BH_velocities = custom_vals['velocity']

        # the above choice depends on how the ics team have implemented it

        ## Load the custom_vals into Class objects
        list_of_BH = []
        for i in range(init_BH_values['N']):
            BH = BlackHole( init_BH_masses[i], init_BH_positions[i], init_BH_velocities[i] )
            list_of_BH.append(BH)

        ## Save the file
        with open('BH_data_ic.pkl', 'wb') as handle:
            pickle.dump(list_of_BH, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        # properly sample init_BH_values (team ICs work)
        ##
        read_from_file = True   # set to False if you want random values

    # for i in range(N_BH):
    #     BH_data_ic.append(Black_hole(init_BH_values))
    # or if they are assigned as separate np arrays, 
    #     BH_data_ic.append(Black_hole(init_BH_masses, init_BH_positions, init_BH_velocities))


# one example of custom values 
## Change the digit format in the array to float
custom_vals = { 'N': 2,
                'mass': np.array([1.0e7, 1.0e7]), 
                'position': np.array([[1., 0., 0.], [-1., 0., 0.]]), 
                'velocity': np.array([[0. ,3.2791 ,0.], [0. ,-3.2791 ,0.]])}

## Calculate the velocities for binary stars
## Set the unit system first
ureg = UnitRegistry()
vel_unit = ureg.km / ureg.s
dist_unit = ureg.kpc
mass_unit = ureg.kg * 1.98892e30
G = 4.301e-3 * (1e-3 * ureg.kpc) / mass_unit * (vel_unit) ** 2 

## Calculate the velocity 
velocity = np.sqrt( G * custom_vals['mass'][0] * mass_unit / dist_unit  ) / vel_unit / 2
# custom_vals['velocity'][0] = np.array([0.,  velocity, 0.])
# custom_vals['velocity'][1] = np.array([0., -velocity, 0.])

# now initialize the black holes with mass, positions, and velocities using the function supplied by the ics team
# N_BH is the number of BHs, BH_data is the list of length N_BH containing BH objects 
BH_data = generate_ICs(N_BH = 2, custom_vals = custom_vals)   

# or load it from some file as
with open('BH_data_ic.pkl', 'rb') as f:
    BH_data_ic = pickle.load(f)

## Test if the data is successfully loaded
# print( BH_data_ic[0].mass, BH_data_ic[0].position, BH_data_ic[0].velocity )

# write the general conic section solution 

# the idea is that in centre of mass frame and relative coordinates (r vector = r_1 vector - r_2 vector), 
# r vector traces out a conic section. The general equation of the conic is 
#                           r(1 + e cos(phi)) = rhs
# where e is eccentricity, r is the magnitude of r vector, phi is the polar angle measured relative to the point where r becomes minimum, 
# and rhs is just some quantity that depends on G, m_1, m_2, energy and angular momentum. (cf. chapter 8 of Classical Mechanics by John R Taylor)
# We just need to verify that this equation holds at all timesteps
# For that we need to first compute the relative r and v vectors  

# the following function takes in the list of BH_objects with their final positions
# and velocities stored as attributes. If the saving is happening in batches and with 
# pickling, all the batches must be unpickled and the list 'BH_data_final' must be constructed

class AnalyticalCheck():
    """
    (docstring to be updated)

    1) converts the positions and velocities into relative coordinates. 
    2) computes the relevant physical quantities
    3) checks if the conic equation above is satisfied at all time steps or not
    4) implicitly checks that the motion is planar as well

    NOTE: It is convenient to demand that the analytical check be run on 2 black holes with initial velocities 
          and positions lying entirely in the x-y plane. This way we can also check for the planarity of motion
          and also convert to polar coordinates easily.  

    input: 
    BH_data_final: a list of BH objects with their final attributes at all timesteps contained within.
                   Assumes that consistency checks for sizes of position and velocity attribute arrays have 
                   been taken care of by the respective team before saving this data. Initial conditions to 
                   generate this must have given positions and velocities with no z component. 
    output:
    TBD exactly 
    """
    
    def __init__(self, BH_data_final):

        if len(BH_data_final) != 2:
            raise ValueError("Analytical solution only exists for the two-body problem. Limit number of black holes to 2.")
        
        self.BH_data_final = BH_data_final
        self.tot_times = np.shape(BH_data_final[0].position)[-1] 
        # assuming the time step stacking to be done such that the last dimension is the number of total timesteps 

#######################################################


#######################################################

    def transform_coords(self):
        """
        transforms the exisiting position and velocity coordinates into a single, relative coordinate
        which appears in the solution to the 2-body problem alongside the reduced mass. 
        """

        r1_vec = self.BH_data_final[0].position  
        r2_vec = self.BH_data_final[1].position
        # both of these will be M x 3 arrays (with the 3rd column being always zero ideally)
        # hereby M is the total number of timesteps (including initial)

        assert r1_vec[:, 2] == 0 and r2_vec[:, 2] == 0, "WARNING: motion of the two bodies deviates from planarity."

        # FIXME - if planar, can do away with the third column 

        relative_r_vec = r1_vec - r2_vec  # M x 3 array
        # remember that relative_r_vec is the same in any frame (CM or not)
    
        v1_vec = self.BH_data_final[0].velocity
        v2_vec = self.BH_data_final[1].velocity   # M x 3 arrays 

        relative_v_vec = v1_vec - v2_vec  # M x 3 array 

        m_1 = self.BH_data_final[0].mass 
        m_2 = self.BH_data_final[1].mass # scalar

        reduced_mass = m_1*m_2/(m_1 + m_2)  # scalar
        # as mentioned in the docstring, no need for dimensional consistency checks on these

        self.relative_r_vec = relative_r_vec
        self.relative_v_vec = relative_v_vec
        self.mu = reduced_mass
        self.m1 = m_1
        self.m2 = m_2

############################################################
        

############################################################

    def compute_energy_mom(self, tol_frac = 1e-3):
        """computes two arrays that give the ang_mom and energy at each timestep, can readily check for conservation
       
       inputs: 
       tol_frac: sets the tolerance fraction for checking the constancy of energy and momentum    
        """
        ang_mom = self.mu * np.linalg.norm(np.cross(self.relative_r_vec, self.relative_v_vec), axis = 1) # axis = 1 implies that self.relative_r_vec and v are Mx3 arrays and not 3xM
        # follows from L = r cross p for the effective 1 body coordinates
        # M x 1 array

        energy = 1/2*self.mu*np.sum(self.relative_v_vec**2, axis = 1) + ang_mom**2/(2*self.mu*np.sum(self.relative_r_vec**2, axis = 1)) - G*self.m1*self.m2/np.linalg.norm(self.relative_r_vec, axis = 1)
        # follows from E = 1/2 mu v^2 + L^2/(2 mu r^2) - G m_1 m2/r for the effective 1 body coordinates
        # M x 1 array

        assert np.shape(ang_mom)[0] == self.tot_times and np.shape(energy)[0] == self.tot_times, "dimensions of energy and ang_momentum data do not match the total number of timesteps"

        # Now check for conservation of both
        if np.all(ang_mom - ang_mom[0] < tol_frac * ang_mom[0]):
            print(f"Angular momentum is conserved at the {np.max(ang_mom/ang_mom[0] - 1)*100:.4f} % level from the starting value")
        else:
            print(f"Angular momentum is NOT conserved at the {tol_frac*100} % level from the starting value")

        if np.all(energy - energy[0] < tol_frac * energy[0]):
            print(f"Energy is conserved at the {np.max(energy/energy[0] - 1)*100:.4f} % level from the starting value")
        else:
            print(f"Energy is NOT conserved at the {tol_frac*100} % level from the starting value")
        # seems like a better option because more explicitly states the level it is conserved at? 

        # assert np.all(ang_mom - ang_mom[0] < tol_frac * ang_mom[0]), f"Angular momentum is not conserved at the {tol_frac*100} % level from the starting value"
        # assert np.all(energy - energy[0] < tol_frac * energy[0]), f"Energy is not conserved at the {tol_frac*100} % level from the starting value"

        self.ang_mom = ang_mom
        self.energy = energy

    ############################################################


    ############################################################

    def trajectory_checker(self):
        """
        makes the more stringent test of checking if the relative coordinates satisfy the general conic equation
        """

        # evaluate rhs (RHS of the equation r(1+cos(phi)) = rhs as described in the comment above this class) 

        rhs = self.ang_mom**2 * (self.m1 + self.m2)/(G * self.m1**2 * self.m2**2)

        eccen_squared = 1 + 2*(self.m1 + self.m2)*self.energy*self.ang_mom**2/(G**2 * self.m1**3 * self.m2**3)
        eccen = np.sqrt(eccen_squared[0])   # conditions at the first time step should ideally determine the dynamics and hence the overall eccentricity 
        self.eccen = eccen

        # we need to measure phi_0 (phi at t=0) since phi = 0 happens when r = r_min (but we are not necessarily starting off at r_min)

        cos_phi_0 = rhs[0]/( self.eccen*np.linalg.norm(self.relative_r_vec[0]) ) - 1
        # phi_0 = np.arccos(cos_phi_0)
        
        # measure the cosine of the increment in phi at all timesteps (=delta_phi) as compared to the starting phi_0
        cos_delta_phi = np.sum(self.relative_r_vec[1:] * self.relative_r_vec[0], axis = 1)/(np.linalg.norm(self.relative_r_vec[1:], axis = 1) * np.linalg.norm(self.relative_r_vec[0]))
        
        sin_phi_0 = np.sqrt(1 - cos_phi_0**2)
        sin_delta_phi = np.sqrt(1 - cos_delta_phi**2)
        # FIXME - what happens after delta_phi becomes larger than pi?

        if np.linalg.norm(self.relative_r_vec[0]) < np.linalg.norm(self.relative_r_vec[1]):
            # r is increasing, phi_0 is positive 
            cos_phi = cos_phi_0 * cos_delta_phi - sin_phi_0 * sin_delta_phi
        else:
            # r is decreasing, phi=0 is yet to come, so phi_0 must be negative
            cos_phi = cos_phi_0 * cos_delta_phi + sin_phi_0 * sin_delta_phi

        # need to check the equality of these two across all timesteps - 

        assert np.allclose(np.linalg.norm(self.relative_r_vec) * (1 + self.eccen * cos_phi), rhs), "conic equation fails at some timestep"
        # both arguments of allclose here are Mx1 arrays
        # can pass a custom tolerance fraction as the input to allclose if needed

###############################################################


###############################################################

    def wrapper_for_analytical_check(self, tol_frac):

        self.transform_coords()
        self.compute_energy_mom(tol_frac = tol_frac)
        self.trajectory_checker()


# Write a plotting function showing both the trajectory of analytical solution and the position of the particle
# Also define a loss function
def Plotting_for_Binary( ics, BH_result ):

    """
    input:
    ics: The example initial conditions for binary stars.
            A list of BlackHole class objects 
            For the binary stars, the
                                

    BH_result: Expected results after simulations
    """

    # Read the initial conditions of the binary stars
    # Radius, Velocity, and the center of the orbit
    R = np.linalg.norm(ics[0].position - ics[1].position) / 2
    V = ics[0].velocity
    Total_Mass = ics[0].mass + ics[1].mass

    COM = np.zeros(3)
    for i in range(2): 
        COM += (ics[i].position * ics[i].mass) / Total_Mass

    
    # Plot a circle with radius R, center at COM based on the case
    fig, ax = plt.subplots(figsize=(8, 8))
    circle = patches.Circle( COM[:2], R, fill=False, linewidth=1, ls = "--")

    ax.add_patch(circle)

    ax.set_xlim( -2*R, 2*R )
    ax.set_ylim( -2*R, 2*R )

    ax.set_xlabel("kpc")
    ax.set_ylabel("kpc")

    plt.show()

    check1 = AnalyticalCheck( BH_result )



# example call - 

# Based on the simulation and the example
ics = BH_data_ic

R = np.linalg.norm(ics[0].position - ics[1].position) / 2
V = np.linalg.norm(ics[0].velocity)
Total_Mass = ics[0].mass + ics[1].mass

COM = np.zeros(3)
for i in range(2): 
    COM += (ics[i].position * ics[i].mass) / Total_Mass

# Generate the analytical results for the given example

# 8 data points

BH_data_pos = np.array([[ics[0].position, ics[1].position]])

phi_1 = np.arccos( np.dot((ics[0].position - COM), np.array([R, 0, 0])))
phi_2 = np.arccos( np.dot((ics[1].position - COM), np.array([R, 0, 0])))

for i in range(1, 101):
    # Calculate the position of two black hole
    phi = i / 100 * 2 * np.pi
    # pos1: position of BlackHole 1
    # pos2: position of BlackHole 2
    pos_1 = np.array([ R*np.cos(phi_1 + phi), R*np.sin(phi_1 + phi), 0.])
    pos_2 = np.array([ R*np.cos(phi_2 + phi), R*np.sin(phi_2 + phi), 0.])

    new_pos = np.array( [[pos_1, pos_2]] )
    BH_data_pos = np.concatenate( (BH_data_pos, new_pos), axis=0 )


# Plot a circle with radius R, center at COM based on the case

fig, ax = plt.subplots(figsize=(8, 8))
circle = patches.Circle( COM[:2], R, fill=False, linewidth=1, ls = "--")
ax.add_patch(circle)
ln, = ax.plot([], [], 'ro')

def init():
    ax.set_xlim( -2*R, 2*R )
    ax.set_ylim( -2*R, 2*R )
    ax.set_xlabel("kpc", fontsize=15)
    ax.set_ylabel("kpc", fontsize=15)
    return ln,

def update(frame):
    xdata = BH_data_pos[frame][:, 0]
    ydata = BH_data_pos[frame][:, 1]
    ln.set_data(xdata, ydata)
    return ln,

ani = FuncAnimation(fig, update, frames=np.arange(100),
                    init_func=init, blit=True)
plt.show()

# check1 = AnalyticalCheck(BH_data_final)
# check1.wrapper_for_analytical_check(tol_frac = 1e-3)
