import warnings
from ..BlackHoles_Struct import BlackHole 
from ..Forces import _comp_acceleration, _comp_jerk, _comp_snap, recalculate_dynamics
from ..evolution import comp_adaptive_dt
from ..ICs import generate_plummer_initial_conditions
from pint import UnitRegistry
from pint import UnitStrippedWarning
import numpy as np

ureg = UnitRegistry()

def test_dynamics(): # Test gravitational forces between black holes earth_like and sun_like
    expected_acceleration = 5.930e-6
    expected_jerk = 1.1805e-12
    expected_snap = 2.3484e-19
    expected_dt = 31.558

    bh1_earth = BlackHole(mass=3.0e-6, position=(ureg("1astronomical_unit").to("kpc").magnitude, 0, 0), velocity=(0, 29.78, 0))  # Earth-like BH
    bh2_sun = BlackHole(mass=1.0, position=(0,0, 0), velocity=(0, 0, 0))  # Sun-like BH

    a = _comp_acceleration(bh1_earth, bh2_sun) # acceleration on earth_like due to sun_like
    j = _comp_jerk(bh1_earth, bh2_sun)
    s = _comp_snap(bh1_earth, bh2_sun, a)
    t = comp_adaptive_dt(a, j, s, eta=0.1, tot_time = ureg("myr").to("s").magnitude)
    print(t)
    # Assertion check
    #assert abs(actual - expected)/expected <= tolerance
    #expected_acceleration = 6e-06
    #expected_acceleration = 5.930262843244524e-06
    accel_error = np.abs(np.linalg.norm(a).magnitude/expected_acceleration - 1)
    jerk_error = np.abs(np.linalg.norm(j).magnitude/expected_jerk - 1)
    snap_error = np.abs(np.linalg.norm(s).magnitude/expected_snap - 1)
    time_step_error = np.abs(t.magnitude/expected_dt - 1)

    assert accel_error < 1e-4, f"Error in acceleration calculation exceeds threshold, error: {accel_error}"  # acceleration magnitude on the earth due to the sun in km/s^2
    assert jerk_error < 1e-4, f"Error in jerk calculation exceeds threshold, error: {jerk_error}"
    assert snap_error < 1e-4, f"Error in snap calculation exceeds threshold, error: {snap_error}"
    assert time_step_error < 1e-4, f"Error in snap calculation exceeds threshold, error: {time_step_error}"
    
def test_tree():
    """
    tests tree calculation of forces against brute force computation
    """
    with warnings.catch_warnings():
        #suppress unit stripping warnings inside this test
        warnings.simplefilter("ignore", category=UnitStrippedWarning)

        blackholes = generate_plummer_initial_conditions(100, 20, 20)[0]['data']

        recalculate_dynamics(blackholes, use_tree=False, use_dynamic_criterion = True, ALPHA = 0.1, THETA_0 = 0.1)

        brute_force_accels = np.asarray([bh.acceleration for bh in blackholes])

        recalculate_dynamics(blackholes, use_tree=True, use_dynamic_criterion= True, ALPHA = 0.1, THETA_0 = 0.1)

        tree_accels = np.asarray([bh.acceleration for bh in blackholes])

    error = np.asarray([(tree_accels[i] - brute_force_accels[i])/brute_force_accels[i] for i in range(len(blackholes))])

    rms_error = np.sqrt(np.sum(error**2) / 3*len(blackholes))

    print(rms_error)
    assert rms_error < 0.01, f"root mean squared error in forces exceeds 1%, error was: {rms_error}"
