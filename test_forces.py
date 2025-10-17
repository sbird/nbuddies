import unittest
import pytest 
from BlackHoles_Struct import BlackHole 
from Forces import _comp_acceleration, recalculate_accelerations
from pint import UnitRegistry
import numpy as np

ureg = UnitRegistry()

def test_Forces(): # Test gravitational forces between black holes earth_like and sun_like
    bh1_earth = (BlackHole(mass=3.0e-6, position=(ureg("1astronomical_unit").to("kpc").magnitude, 0, 0), velocity=(0, 0, 0)))  # Earth-like BH
    bh2_sun = (BlackHole(mass=1.0, position=(0,0, 0), velocity=(0, 0, 0)))  # Sun-like BH

    a = _comp_acceleration(bh1_earth, bh2_sun) # acceleration on earth_like due to sun_like
    # Assertion check
    #assert abs(actual - expected)/expected <= tolerance
    expected_acceleration = 6e-06
    #5.930262843244524e-06
    #assert np.isclose(np.linalg.norm(a).magnitude, expected_acceleration, atol=1e-7)  
    assert np.linalg.norm(a).magnitude ==  expected_acceleration  # acceleration magnitude on the earth due to the sun in km/s^2

       





