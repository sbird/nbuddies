from pint import UnitRegistry
from BlackHoles_Struct import BlackHole
from ICs import generate_initial_conditions
from evolution2 import simulation
import os
import numpy as np 
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

## Set the unit system
ureg = UnitRegistry()
vel_unit = ureg.km / ureg.s
dist_unit = ureg.kpc
mass_unit = ureg.kg * 1.98892e30
G = 4.301e-3 * (1e-3 * ureg.kpc) / mass_unit * (vel_unit) ** 2 

## Test section for generating analytical dataset
def generate_analy_dataset( BH_data_ic ):
    # Based on the simulation and the example
    ics = BH_data_ic['data']

    R = np.linalg.norm(ics[0].position - ics[1].position) / 2
    V = np.linalg.norm(ics[0].velocity)
    Total_Mass = ics[0].mass + ics[1].mass

    COM = np.zeros(3)
    for i in range(2): 
        COM += (ics[i].position * ics[i].mass) / Total_Mass

    # Generate the analytical results for the given example

    # 100 data points

    BH_data_pos = np.array([[ics[0].position, ics[1].position]])
    BH_data_vel = np.array([[ics[0].velocity, ics[1].velocity]])

    phi_1 = np.arccos( np.dot((ics[0].position - COM), np.array([R, 0, 0])))
    phi_2 = np.arccos( np.dot((ics[1].position - COM), np.array([R, 0, 0])))

    for i in range(1, 101):
        # Calculate the position of two black hole for 100 data points divided equally on the circle
        # position = R [ cos(phi), sin(phi), 0]
        # velocity = V [-sin(phi), cos(phi), 0]
        phi = i / 100 * 2 * np.pi
        # pos1: position of BlackHole 1
        # pos2: position of BlackHole 2
        pos_1 = np.array([ R*np.cos(phi_1 + phi), R*np.sin(phi_1 + phi), 0.])
        pos_2 = np.array([ R*np.cos(phi_2 + phi), R*np.sin(phi_2 + phi), 0.])
        # vel1: velocity of BlackHole 1
        # vel2: velocity of BlackHole 2
        vel_1 = np.array([ -V*np.sin(phi_1 + phi), V*np.cos(phi_1 + phi), 0.])
        vel_2 = np.array([ -V*np.sin(phi_2 + phi), V*np.cos(phi_2 + phi), 0.])
        # Append the data together into the structure [[pos1_t1, pos2_t1], [pos1_t2, pos2_t2], ...]
        new_pos = np.array( [[pos_1, pos_2]] )
        new_vel = np.array( [[vel_1, vel_2]] )
        BH_data_pos = np.concatenate( (BH_data_pos, new_pos), axis=0 )
        BH_data_vel = np.concatenate( (BH_data_vel, new_vel), axis=0 )
    
    # Save the data into files
    n_files = np.shape(BH_data_pos)[0]
    N = 2

    ## Load the custom_vals into Class objects
    # Save the data with the structure [[BH1_t0, BH2_t0], [BH1_t1, BH2_t1], ...]
    list_of_BH = []
    for i in range(n_files):
        for j in range(N):
            BH = BlackHole( ics[j].mass, BH_data_pos[i][j], BH_data_vel[i][j] )
            list_of_BH.append(BH)

        ## Save the file
        # If the directory does not exist, create it
        Save_Dir = "./data_test/"
        if not os.path.exists(Save_Dir):
            os.makedirs(Save_Dir)
        with open( Save_Dir + 'BH_data_%03d.pkl' % i, 'wb') as handle:
            pickle.dump([list_of_BH], handle, protocol=pickle.HIGHEST_PROTOCOL)
    
def generate_binary_ICs(N_BH, custom_vals = None, analy_sets = False):
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

        data = dict()
        data['data'] = list_of_BH
        ## Save the file
        with open('BH_data_ic.pkl', 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # If required, generate the analytical dataset for this binary system
        if analy_sets:
            generate_analy_dataset( data )

# example call - 
# one example of custom values 
## Change the digit format in the array to float
custom_vals = { 'N': 2,
                'mass': np.array([1.0e7, 1.0e7]), 
                'position': np.array([[1., 0., 0.], [-1., 0., 0.]]), 
                'velocity': np.array([[0. ,3.2791 ,0.], [0. ,-3.2791 ,0.]])}

## Test the velocities for binary stars
r = np.linalg.norm( custom_vals['position'][1] - custom_vals['position'][0] )
velocity = np.sqrt( G * custom_vals['mass'][0] * mass_unit / (2 * r * dist_unit)  ) / vel_unit

assert np.isclose( velocity.magnitude , 3.2791, atol=1e-4), "Velocity in custom_vals error"

# now initialize the black holes with mass, positions, and velocities using the function supplied by the ics team
# N_BH is the number of BHs, BH_data is the list of length N_BH containing BH objects 
BH_data = generate_binary_ICs(N_BH = 2, custom_vals = custom_vals, analy_sets = False)   

# or load it from some file as
with open('BH_data_ic.pkl', 'rb') as f:
    BH_data_ic = pickle.load(f)

ICS_path = "./BH_data_ic.pkl"
output_dir = "./data/"

# Implement the evolution code here
Total_time = 5*10**17               # Total evoultion time in seconds
n_snapshots = 100                   # Number of the output snapshots
delta_t_fraction = n_snapshots      # How many steps between two snapshots
                                    # Due to the issue in output functions, set this to be the same as n_snapshots 
                                    # to get an expected output files

# Run the simulation here
simulation( ICS_path, output_dir, Total_time, Total_time // n_snapshots // delta_t_fraction, Total_time // n_snapshots)

# Plot a circle with radius R, center at COM based on the case
COM = [0,0]
R = np.linalg.norm(custom_vals['position'][0] - custom_vals['position'][1]) / 2

fig, ax = plt.subplots(figsize=(8, 8))
circle = patches.Circle( COM, R, fill=False, linewidth=1, ls = "--")
ax.add_patch(circle)
ln, = ax.plot([], [], 'ro')

# Create a text object to display the loss
loss_text = ax.text(0.05, 0.95, '', transform=ax.transAxes,
                    fontsize=14, va='top', ha='left',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

# Initialize the plot limits and labels
def init():
    ax.set_xlim( -2*R, 2*R )
    ax.set_ylim( -2*R, 2*R )
    ax.set_xlabel("kpc", fontsize=15)
    ax.set_ylabel("kpc", fontsize=15)
    loss_text.set_text('')  # Clear the loss text
    return ln,

# Update the loss function defined as the fraction of the distance from the center and the radius
def loss_func( xdata, ydata, R ):
    loss = 0.
    for i in range(len(xdata)):
        loss += ((np.sqrt( xdata[i]**2 + ydata[i]**2 ) - R) / R)
    return loss

# Update the plot and make it animated
def update(frame):
    # Load the corresponding snapshot
    with open( output_dir + 'data_batch' + str(n_snapshots) + '.pkl', 'rb') as f:
        BH_data_final = pickle.load(f)

    xdata = []
    ydata = []
    for frame in range(frame + 1):
        for i in range(2):
            xdata.append(BH_data_final[frame][i].position[0])
            ydata.append(BH_data_final[frame][i].position[1])

    xdata = xdata[-2:]
    ydata = ydata[-2:]

    ln.set_data(xdata, ydata)
    loss = loss_func(xdata, ydata, R)
    loss_text.set_text(f'Loss: {loss:.4f}')
    return ln, loss_text

ani = FuncAnimation(fig, update, frames=np.arange(n_snapshots),
                    init_func=init, blit=True)
ani.save('binary_simulation.gif', fps=5)
