
#N-Body Gravitational Forces Calculations

import numpy as np
from pint import UnitRegistry 
from BlackHoles_Struct import BlackHole
from gravitree import Node, build_tree

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

#Tree accuracy parameters
ALPHA = 1e-4
THETA_0 = 0.1

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
def _comp_snap(BH_i: BlackHole, BH_j: BlackHole, previous_accel_i : list[float]):
    """
    Computes the snap value between two blackholes using the analytical expression for snap.
    s_ij= G * m_j [(a_ij/r_ij**3)- (6 * v_ij*(r_ij dot v_ij)/ r_ij**5) - (3 * r_ij * (v_ij**2 + r_ij dot a_ij)/ r_ij **5) + (15 * (r_i*(r_ij dot v_ij)**2)/ r_ij**7)]
    Returns snap in units km/s^4

    Note whether or not BH_j.accel is the new or old accel depends on if BH_j has had its accel computed before or after BH_i. I don't think this will cause issues, but something to be aware of.
    
    Parameters
    ----------
    BH_i : Blackhole
        blackhole whose snap is being computed
    BH_j : Blackhole
        source of the snap
    previous_accel_i : list[float]
        acceleration of BH_i at last timestep
    
    Returns
    -------
    list[float]
        snap of BH_i due to BH_j
    """

    #in first time step accels may not have units, handles that here
    if not hasattr(BH_j.acceleration, 'units'):
        BH_j.acceleration = BH_j.acceleration * ureg('km/s^2')
    if not hasattr(previous_accel_i, 'units'):
        previous_accel_i = previous_accel_i * ureg('km/s^2')

    r_ij = BH_i.displacement(BH_j) * ureg.kpc   #Calls position vector function from BlackHoles_Struct.py
    v_ij = (BH_j.velocity - BH_i.velocity) * ureg('km/s') #Difference in velocities 
    a_ij = BH_j.acceleration - previous_accel_i #Difference in accelerations

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

def recalculate_dynamics(BHs: list[BlackHole], use_tree : bool = True):
    """
    Function for looping over black holes 
    Recalculates acceleration jerk and snap
    
    Parameters
    ----------
    BHs : list[BlackHole]
        the blackholes to have their dynamical values
    use_tree : bool, default False
        whether or not to use the tree for force calculation
    """
    if use_tree:
        root = build_tree(BHs)

    for BH_i in BHs: 
        #Reset acceleration before summing 
        previous_accel_i = BH_i.acceleration
        BH_i.acceleration = np.zeros(3)
        BH_i.jerk = np.zeros(3)
        BH_i.snap = np.zeros(3) 

        #Tree calculation
        if use_tree:
            _calculate_accel_with_tree(BH_i, root, previous_accel_i)
            continue

        #brute force calculation
        for BH_j in BHs: 
            if BH_i == BH_j: 
                continue
            
            BH_i.acceleration += _comp_acceleration(BH_i, BH_j) 
            BH_i.jerk += _comp_jerk(BH_i, BH_j)
            BH_i.snap += _comp_snap(BH_i, BH_j, previous_accel_i)

def _calculate_accel_with_tree(bh : BlackHole, root : Node, previous_accel : list[float]):
    """
    Recursively calculates acceleration with tree

    Parameters
    ----------
    bh : BlackHole
        balckhole whose acceleration is being computed
    root : Node
        root node of tree from which accleration is being computed
    previous_accel : list[float]
        acceleration of bh at previous timestep
    """
    #ignore empty nodes
    if len(root.enclosed_blackholes) == 0:
        return
    
    #ignore nodes that only contain self
    if len(root.enclosed_blackholes) == 1 and root.enclosed_blackholes[0] == bh:
        return
    
    #immediately compute nodes with only 1 BH
    if len(root.enclosed_blackholes) == 1:
        bh.acceleration += _comp_acceleration(bh, root)
        bh.jerk += _comp_jerk(bh, root)
        bh.snap += _comp_snap(bh, root, previous_accel)
        return
    
    #compute nodes which meet approximation criterion
    if node_is_approximatable(bh, root, np.linalg.norm(previous_accel)) and not root.is_inside(bh.position):
        bh.acceleration += _comp_acceleration(bh, root)
        bh.jerk += _comp_jerk(bh, root)
        bh.snap += _comp_snap(bh, root, previous_accel)
        return
    
    #descend tree further as node did not meet any end criterion
    for child in root.children:
        _calculate_accel_with_tree(bh, child, previous_accel)
    
def node_is_approximatable(bh : BlackHole, node : Node, previous_accel : float, use_dynamic_criterion : bool = True):
    """
    Determines if Node is safe to approximate or if tree needs to be further explored

    Parameters
    ----------
    bh : Blackhole
        the blackhole who's accel is being calculated
    node : Node
        the node whos approximation status is in question
    previous_accel : float
        magnitude of acceleration at previous timestep, only needed for dynamic criterion
    use_dynamic_criterion : bool
        determines if geometric criterion (default) or dynamic criterion should be used

    Returns
    bool
        if the node is sufficiently far to be approximated as a point mass
    """
    #dynamic criterion
    if use_dynamic_criterion:
        d = np.linalg.norm(bh.displacement(node)) * ureg.kpc
        return GG * node.mass * ureg.solarmass * (node.crossection * ureg.kpc)**2/(d**4) <= ALPHA * previous_accel
    
    #geometric criterion
    return (node.crossection / np.linalg.norm(bh.displacement(node))) < THETA_0