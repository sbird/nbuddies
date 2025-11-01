from binary_analytical import AnalyticalCheck
import numpy as np
from pint import UnitRegistry
from evolution2 import simulation
from binary import generate_binary_ICs

## Set the unit system first
ureg = UnitRegistry()
vel_unit = ureg.km / ureg.s
dist_unit = ureg.kpc
mass_unit = ureg.kg * 1.98892e30
G = 4.301e-3 * (1e-3 * ureg.kpc) / mass_unit * (vel_unit) ** 2 

def test_binary():

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

    # Path to the output data directory
    ICS_path = "./BH_data_binary.pkl"
    output_dir = "./data/"

    # Implement the evolution code here
    Total_time = 5*10**17               # Total evoultion time in seconds
    n_snapshots = 100                   # Number of the output snapshots

    # Run the simulation here
    simulation(ICS_path, output_dir, Total_time, n_snapshots, delta_t = None, adaptive_dt= True, eta = 0.1, use_tree = True )

    # Initialize the AnalyticalCheck object
    analytical_solution = AnalyticalCheck(output_dir)

    ## The value of angular momentum and energy in our test sample
    # Two BH with 10^7 solar mass each, initial separation 2 kpc, initial velocity 3.2791 km/s
    angular_momentum = 1e7 * 2 * 3.2791 * mass_unit * ureg.kpc * ureg.km / ureg.s   # In Msun * kpc * km/s
    energy = -1.0755e8 * mass_unit * (ureg.km / ureg.s) ** 2                        # In Msun * (km/s)^2

    # Output the computed angular momentum and energy with total tolerance fraction of 0.1%
    ans = analytical_solution.compute_energy_mom(tol_frac = 1e-3)

    ## Known results from analytical calculation
    # Convert analytical results to consistent units
    angular_momentum = angular_momentum.to("kg * km**2 / s")
    energy = energy.to("kg * (km/s)**2")

    ## Computed results from the simulation
    # Convert computed results to consistent units
    comp_angular_momentum = ans[0].to("kg * km**2 / s")
    comp_energy = ans[1].to("kg * (km/s)**2")

    # Assert that the computed values are within the specified tolerance of the analytical values
    assert np.all(abs(comp_angular_momentum.m - angular_momentum.m) / angular_momentum.m < 1e-3), "Angular momentum does not match analytical solution within tolerance"
    assert np.all(abs((comp_energy.m - energy.m) / energy.m) < 1e-3), "Energy does not match analytical solution within tolerance"
