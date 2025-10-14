#N-Body Data Structure
import numpy as np

class BlackHole():
    """
    Define data structure --> struct 
    mass, position (3D), velocity (3D), and acceleration (initialized to zero)
    Validate data before assigning them
    """
    def __init__(self, mass: float, position: list[float], velocity: list[float]):
        assert mass > 0, "Mass must be positive" #checks mass is positive 
        assert len(position) == 3, "Position must be a 3D vector" #checks that position is a vector
        assert len(velocity) == 3, "Velocity must be a 3D vector" #checks that velocity is a vector
       
        self.mass = mass 
        self.position = np.array(position) 
        self.velocity = np.array(velocity) 
        self.acceleration = np.zeros(3) 
        

#Displacement and equality functions must go in the class because self will not work outside of class

    def displacement(self, other):
        """
        Computes displacement vector from self to another black hole
        """
        return other.position - self.position 

    def __eq__(self, other): 
        """
        Compare all parameters except acceleration to check if two BHs are the same 
        """
        if other.mass != self.mass: 
            return False
        for i in range(3):
            if other.position[i] != self.position[i]: 
                return False
            if other.velocity[i] != self.velocity[i]: 
                return False 
        return True 