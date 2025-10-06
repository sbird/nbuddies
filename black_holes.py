import numpy as np

class Black_hole:
    def __init__(self, mass: float, position: list[float], velocity: list[float]):
        """
        Initialized Black Hole, sets mass and intital position, velocity, and acceleration.

        Parameters
        ----------
        mass : float
            mass of black hole
        position : list[float]
            position vector of black hole in kpc
        velocity : list[float])
            velocity vector of black hole in km/s
        """
        self.mass = mass
        self.position = np.array(position)
        self.velocity = np.array(velocity)
        self.acceleration = np.zeros(3)

    def __eq__(self, other):
        """
        Checks if two blacks holes are the same by checking all except acceleration.

        Parameters
        ----------
        other : Black_hole
            The black hole being compared to

        Returns
        -------
        bool
            Whether or not these black holes are the same.
        """
        if other.mass != self.mass:
            return False
        
        for i in range(3):
            if other.position[i] != self.position[i]:
                return False
            
            if other.velocity[i] != self.velocity[i]:
                return False
            
        return True

