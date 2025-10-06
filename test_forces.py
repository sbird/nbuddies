from forces import *


def test_2_body_grav_calc():
    GG = 4.301e-6 # Newton constant km^2 kpc / Msun s^2
    BH_1 = Black_hole(10*np.random.rand(), [10*np.random.rand(), 10*np.random.rand(), 10*np.random.rand()], [10*np.random.rand(), 10*np.random.rand(), 10*np.random.rand()])
    BH_2 = Black_hole(10*np.random.rand(), [10*np.random.rand(), 10*np.random.rand(), 10*np.random.rand()], [10*np.random.rand(), 10*np.random.rand(), 10*np.random.rand()])

    data = [BH_1, BH_2]
    recalculate_acceleration_due_to_gravity(data)

    displacement = BH_2.position - BH_1.position
    accel_1 = GG * BH_2.mass * (displacement) / np.pow(displacement[0]*displacement[0] + displacement[1]*displacement[1] + displacement[2]*displacement[2], 3/2)
    
    print(BH_1.acceleration)
    print(accel_1)
    assert np.prod(np.isclose(BH_1.acceleration, accel_1)), "2 body gravity calculation is wrong"