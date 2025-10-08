from black_holes import *

def recalculate_acceleration_due_to_gravity(data: list[Black_hole]):
    """
    Recalculates accelerations of all black holes

    Parameters
    ----------
    data : list[Black_hole]
        list of black holes to be target and basis of acceleration recalculation
    """
    for BH_i in data:
        BH_i.acceleration = np.zeros(3)
        for BH_j in data:
            if BH_i == BH_j:
                continue
            BH_i.acceleration += calculate_acceleration_from_one_body(BH_i, BH_j)

def calculate_acceleration_from_one_body(target: Black_hole, source: Black_hole):
    """
    Calculate the accelation of one black hole due to another. Note the units of acceleration are km^2 / kpc s^2

    Parameters
    ----------
    target : Black_hole
        The black hole whose acceleration is being calculated
    source : Black_hole
        The black hole whose gravity is the source of the acceleration
    """
    GG = 4.301e-6 # Newton constant km^2 kpc / Msun s^2
    displacement = source.position - target.position
    return GG * source.mass * displacement / (vector_magnitude(displacement) ** 3)

def vector_magnitude(vec: list[float]):
    """
    Computes magnitude of a vector.

    Parameters
    ----------
    vec : list[float]
        The vectors to have it's magnitude calculated
    
    Returns
    -------
    float
        The magnitude of the vector
    """
    return np.sqrt(np.sum(vec*vec))
