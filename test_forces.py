import warnings
from BlackHoles_Struct import BlackHole 
from Forces import _comp_acceleration, recalculate_dynamics
from ICs import generate_plummer_initial_conditions
from pint import UnitRegistry
from pint import UnitStrippedWarning
import numpy as np

ureg = UnitRegistry()

def test_Forces(): # Test gravitational forces between black holes earth_like and sun_like
    bh1_earth = (BlackHole(mass=3.0e-6, position=(ureg("1astronomical_unit").to("kpc").magnitude, 0, 0), velocity=(0, 0, 0)))  # Earth-like BH
    bh2_sun = (BlackHole(mass=1.0, position=(0,0, 0), velocity=(0, 0, 0)))  # Sun-like BH

    a = _comp_acceleration(bh1_earth, bh2_sun) # acceleration on earth_like due to sun_like
    # Assertion check
    #assert abs(actual - expected)/expected <= tolerance
    #expected_acceleration = 6e-06
    expected_acceleration = 5.930e-6
    #expected_acceleration = 5.930262843244524e-06
    error = np.abs(np.linalg.norm(a).magnitude/expected_acceleration - 1)
    assert error < 1e-4, f"Error in forces calculation exceeds threshold, error: {error}"  # acceleration magnitude on the earth due to the sun in km/s^2

def test_tree():
    """
    tests tree calculation of forces against brute force computation
    """
    with warnings.catch_warnings():
        #suppress unit stripping warnings inside this test
        warnings.simplefilter("ignore", category=UnitStrippedWarning)

        blackholes = generate_plummer_initial_conditions(100, 20, 20)[0]['data']

        recalculate_dynamics(blackholes, use_tree=False)

        brute_force_accels = np.asarray([bh.acceleration for bh in blackholes])

        recalculate_dynamics(blackholes, use_tree=True)

        tree_accels = np.asarray([bh.acceleration for bh in blackholes])

    error = np.asarray([(tree_accels[i] - brute_force_accels[i])/brute_force_accels[i] for i in range(len(blackholes))])

    rms_error = np.sqrt(np.sum(error**2) / 3*len(blackholes))

    print(rms_error)
    assert rms_error < 0.01, f"root mean squared error in forces exceeds 1%, error was: {rms_error}"
