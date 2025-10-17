
#N-Body Gravitational Forces Calculations

import numpy as np
from pint import UnitRegistry 
from BlackHoles_Struct import BlackHole

"""
Initialize unit registry and define units
We decide that the gravitational constant units will be:
    kiloparsec * kilometer^2 / (solarmass * second^2)
Therefore acceleration units of kilometer^2 / (kiloparsec * second^2) (messy for timestep team)
We convert acceleration to km/s^2
"""
ureg = UnitRegistry()
ureg.define('solarmass = 1.98847e30 * kilogram') 
G = 6.67430e-11 * ureg.meter**3 / (ureg.kilogram * ureg.second**2) 
GG = G.to("kiloparsec * kilometer**2 / (solarmass * second**2)") 



def _comp_acceleration(BH_i : BlackHole, BH_j : BlackHole):
    """ 
    Acceleration Function: Compute acceleration on BH_i due to BH_j
    -BH_i = blackhole on which the force is exerted
    -BH_j = blackhole exerting grav force 
    In this function we...
    - Calulate the position vector 
    - Calculate the vector magnitude --> np.linalg.norm() function calculates the magnitude vector (Euclidean norm)
    - Calculate the acceleration using our gravitational constant and converts to km/s**2
    Returns np.ndarray with acceleration vector for BH_i in km/s**2
    """
    pos_vec = BH_i.displacement(BH_j) * ureg.kpc
    mag_vec = (np.linalg.norm(pos_vec.magnitude)) * ureg.kpc
    assert mag_vec.magnitude != 0, "BHs cannot be at the same position - division by zero"
    accel = GG * BH_j.mass * ureg.solarmass * (pos_vec) / (mag_vec ** 3)
    accel = accel.to("km/s**2")  
    return accel

def recalculate_accelerations(BHs: list[BlackHole]):
    """
    Function for looping over black holes 
    Recalculates acceleration and adds it to the existing acceleration (which should be zero)
    """
    for BH_i in BHs: 
        BH_i.acceleration = np.zeros(3) #Reset acceleration before summing 
        for BH_j in BHs: 
            if BH_i == BH_j: 
                continue
            else:
                BH_i.acceleration += _comp_acceleration(BH_i, BH_j) 