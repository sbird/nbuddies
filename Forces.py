
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
    mag_vec = np.linalg.norm(pos_vec) 
    assert mag_vec != 0, "BHs cannot be at the same position - division by zero" #Checks that the magnitude vector is not zero (i.e. same BHs)
    accel = GG * BH_j.mass * ureg.solarmass * (pos_vec)/(mag_vec)**3 
    accel = accel.to("km/s**2")
    return accel            #Converts acceleration to km/s^2

def _comp_jerk(BH_i : BlackHole, BH_j : BlackHole):
    """
    Computes the jerk value between two blackholes using the analytical expression for jerk.
    j_ij = G * m_j [(v_ij/ r_ij **3) - 3(r_ij dot v_ij)*r_ij/ r**5]
    Returns jerk in units units of km/s^3 
    """
    r_ij = BH_i.displacement(BH_j) * ureg.kpc        # Calls dposition vector function function from BlackHoles_Struct.py
    v_ij = (BH_j.velocity - BH_i.velocity) * ureg('km/s')    # Difference in velocities (make sure that it is j minus i for calculation on i)
    r = np.linalg.norm(r_ij) #Magnitude 
    assert r != 0, "Black holes cannot be the same" #If the BHs are the same  
    rdotv = np.dot(r_ij, v_ij) #Dot product of position and velocity vectors
    jerk = GG * BH_j.mass * ureg.solarmass * (v_ij / r**3 - 3 * rdotv * r_ij / r**5) 
    return jerk.to("km/s^3")


#Function to compute snap from jerk
def _comp_snap(BH_i: BlackHole, BH_j: BlackHole):
    """
    Computes the snap value between two blackholes using the analytical expression for snap.
    s_ij= G * m_j [(a_ij/r_ij**3)- (6 * v_ij*(r_ij dot v_ij)/ r_ij**5) - (3 * r_ij * (v_ij**2 + r_ij dot a_ij)/ r_ij **5) + (15 * (r_i*(r_ij dot v_ij)**2)/ r_ij**7)]
    Returns snap in units km/s^4
    """

    r_ij = BH_i.displacement(BH_j) * ureg.kpc   #Calls position vector function from BlackHoles_Struct.py
    v_ij = (BH_j.velocity - BH_i.velocity) * ureg('km/s') #Difference in velocities 
    a_ij = BH_j.acceleration - BH_i.acceleration #Difference in accelerations

    r = np.linalg.norm(r_ij) #Magnitude 
    assert r != 0, "Black holes cannot be the same" #If the BHs are the same  

    rdotv = np.dot(r_ij, v_ij)#Dot product of position and velocity vectors
    v2 = np.dot(v_ij, v_ij) 
    rdota = np.dot(r_ij, a_ij)#Dot product of position and acceleration vectors

    snap = GG * BH_j.mass * ureg.solarmass * (a_ij / r**3 
                                              - 6 * rdotv * v_ij / r**5  
                                              - 3 * (v2 + rdota) * r_ij / r**5  
                                              + 15 * (rdotv**2) * r_ij / r**7 )
    return snap.to('km/s^4')



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


#Then similar to the acceleration function we will need for loops for recalculating jerk and snap
def recalculate_jerks(BHs: list[BlackHole]):
    """
    Function for looping over black holes 
    Recalculates jerk and adds it to the existing jerk (which should be zero)
    """
    for BH_i in BHs:
        BH_i.jerk = np.zeros(3) #Reset jerk before summing 
        for BH_j in BHs:
            if BH_i == BH_j: #Ensure BHs are not the same 
                continue
            BH_i.jerk += _comp_jerk(BH_i, BH_j)


def recalculate_snaps(BHs: list[BlackHole]):
    """
    Function for looping over black holes 
    Recalculates snap and adds it to the existing snap (which should be zero)
    """
    for BH_i in BHs:
        BH_i.snap = np.zeros(3) #Reset snap before summing 
        for BH_j in BHs:
            if BH_i == BH_j: #Ensure BHs are not the same
                continue
            BH_i.snap += _comp_snap(BH_i, BH_j)

## FIXME - can probably combine recalculate snaps, jerk and acceleration into a single loop and function. 