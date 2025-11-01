from binary_analytical import AnalyticalCheck
import numpy as np
from pint import UnitRegistry
from evolution2 import simulation

## Set the unit system first
ureg = UnitRegistry()
vel_unit = ureg.km / ureg.s
dist_unit = ureg.kpc
mass_unit = ureg.kg * 1.98892e30
G = 4.301e-3 * (1e-3 * ureg.kpc) / mass_unit * (vel_unit) ** 2 

def test_binary():
    # Path to the output data directory
    ICS_path = "./BH_data_ic.pkl"
    output_dir = "./data/"

    # Implement the evolution code here
    Total_time = 5*10**17               # Total evoultion time in seconds
    n_snapshots = 100                   # Number of the output snapshots
    # delta_t_fraction = n_snapshots      # How many steps between two snapshots
                                        # Due to the issue in output functions, set this to be the same as n_snapshots 
                                        # to get an expected output files

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
