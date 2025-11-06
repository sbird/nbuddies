import numpy as np 
from pint import UnitRegistry 
from ..Forces import _comp_acceleration, ureg
from ..BlackHoles_Struct import BlackHole
import pickle

'''
Function to test Forces team acceleration units which should be in km/s**2
'''
def test_acceleration_units():
    #create two arbitrary black holes (their values don't matter) and matches BH class format
    BH1 = BlackHole(mass=10, position=[0,0,0], velocity=[0,0,0])
    BH2 = BlackHole(mass=20, position=[1,0,0], velocity=[0,0,0])
    
    #uses the comp acceleration function on the black holes
    result = _comp_acceleration(BH1, BH2)
    #Force team had their wanted units to be km/s**2
    expected_unit = ureg("km/s**2")
    
    #makes sure our wanted units match our actual units from comp acceleration 
    assert result.units == expected_unit, f"Expected {expected_unit}, got {result.units}"