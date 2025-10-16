import numpy as np 
from pint import UnitRegistry
from BlackHoles_Struct import BlackHole

## Set the unit system first
ureg = UnitRegistry()
vel_unit = ureg.km / ureg.s
dist_unit = ureg.kpc
mass_unit = ureg.kg * 1.98892e30
G = 4.301e-3 * (1e-3 * ureg.kpc) / mass_unit * (vel_unit) ** 2 

# write the general conic section solution 

# the idea is that in centre of mass frame and relative coordinates (r vector = r_1 vector - r_2 vector), 
# r vector traces out a conic section. The general equation of the conic is 
#                           r(1 + e cos(phi)) = rhs
# where e is eccentricity, r is the magnitude of r vector, phi is the polar angle measured relative to the point where r becomes minimum, 
# and rhs is just some quantity that depends on G, m_1, m_2, energy and angular momentum. (cf. chapter 8 of Classical Mechanics by John R Taylor)
# We just need to verify that this equation holds at all timesteps
# For that we need to first compute the relative r and v vectors  

# the following function takes in the list of BH_objects with their final positions
# and velocities stored as attributes. If the saving is happening in batches and with 
# pickling, all the batches must be unpickled and the list 'BH_data_final' must be constructed

class AnalyticalCheck():
    """
    (docstring to be updated)

    1) converts the positions and velocities into relative coordinates. 
    2) computes the relevant physical quantities
    3) checks if the conic equation above is satisfied at all time steps or not
    4) implicitly checks that the motion is planar as well

    NOTE: It is convenient to demand that the analytical check be run on 2 black holes with initial velocities 
          and positions lying entirely in the x-y plane. This way we can also check for the planarity of motion
          and also convert to polar coordinates easily.  

    input: 
    BH_data_final: a list of BH objects with their final attributes at all timesteps contained within.
                   Assumes that consistency checks for sizes of position and velocity attribute arrays have 
                   been taken care of by the respective team before saving this data. Initial conditions to 
                   generate this must have given positions and velocities with no z component. 
    output:
    TBD exactly 
    """
    
    def __init__(self, BH_data_final):

        if len(BH_data_final) != 2:
            raise ValueError("Analytical solution only exists for the two-body problem. Limit number of black holes to 2.")
        
        self.BH_data_final = BH_data_final
        self.tot_times = np.shape(BH_data_final[0].position)[-1] 
        # assuming the time step stacking to be done such that the last dimension is the number of total timesteps 

#######################################################


#######################################################

    def transform_coords(self):
        """
        transforms the exisiting position and velocity coordinates into a single, relative coordinate
        which appears in the solution to the 2-body problem alongside the reduced mass. 
        """

        r1_vec = self.BH_data_final[0].position  
        r2_vec = self.BH_data_final[1].position
        # both of these will be M x 3 arrays (with the 3rd column being always zero ideally)
        # hereby M is the total number of timesteps (including initial)

        assert r1_vec[:, 2] == 0 and r2_vec[:, 2] == 0, "WARNING: motion of the two bodies deviates from planarity."

        # FIXME - if planar, can do away with the third column 

        v1_vec = self.BH_data_final[0].velocity
        v2_vec = self.BH_data_final[1].velocity   # M x 3 arrays 

        # as mentioned in the docstring, no need for dimensional consistency checks on these
        # remember that relative_r_vec is the same in any frame (CM or not)
        self.relative_r_vec = v1_vec - v2_vec  # M x 3 array 
        self.relative_v_vec = r1_vec - r2_vec  # M x 3 array
        self.m1 = self.BH_data_final[0].mass # scalar
        self.m2 = self.BH_data_final[1].mass # scalar
        self.mu = self.m1*self.m2/(self.m1 + self.m2)  # scalar

############################################################
        

############################################################

    def compute_energy_mom(self, tol_frac = 1e-3):
        """computes two arrays that give the ang_mom and energy at each timestep, can readily check for conservation
       
       inputs: 
       tol_frac: sets the tolerance fraction for checking the constancy of energy and momentum    
        """
        ang_mom = self.mu * np.linalg.norm(np.cross(self.relative_r_vec, self.relative_v_vec), axis = 1) # axis = 1 implies that self.relative_r_vec and v are Mx3 arrays and not 3xM
        # follows from L = r cross p for the effective 1 body coordinates
        # M x 1 array

        energy = 1/2*self.mu*np.sum(self.relative_v_vec**2, axis = 1) + ang_mom**2/(2*self.mu*np.sum(self.relative_r_vec**2, axis = 1)) - G*self.m1*self.m2/np.linalg.norm(self.relative_r_vec, axis = 1)
        # follows from E = 1/2 mu v^2 + L^2/(2 mu r^2) - G m_1 m2/r for the effective 1 body coordinates
        # M x 1 array

        assert np.shape(ang_mom)[0] == self.tot_times and np.shape(energy)[0] == self.tot_times, "dimensions of energy and ang_momentum data do not match the total number of timesteps"

        # Now check for conservation of both
        if np.all(ang_mom - ang_mom[0] < tol_frac * ang_mom[0]):
            print(f"Angular momentum is conserved at the {np.max(ang_mom/ang_mom[0] - 1)*100:.4f} % level from the starting value")
        else:
            print(f"Angular momentum is NOT conserved at the {tol_frac*100} % level from the starting value")

        if np.all(energy - energy[0] < tol_frac * energy[0]):
            print(f"Energy is conserved at the {np.max(energy/energy[0] - 1)*100:.4f} % level from the starting value")
        else:
            print(f"Energy is NOT conserved at the {tol_frac*100} % level from the starting value")
        # seems like a better option because more explicitly states the level it is conserved at? 

        # assert np.all(ang_mom - ang_mom[0] < tol_frac * ang_mom[0]), f"Angular momentum is not conserved at the {tol_frac*100} % level from the starting value"
        # assert np.all(energy - energy[0] < tol_frac * energy[0]), f"Energy is not conserved at the {tol_frac*100} % level from the starting value"

        self.ang_mom = ang_mom
        self.energy = energy

    ############################################################


    ############################################################

    def trajectory_checker(self):
        """
        makes the more stringent test of checking if the relative coordinates satisfy the general conic equation
        """

        # evaluate rhs (RHS of the equation r(1+cos(phi)) = rhs as described in the comment above this class) 

        rhs = self.ang_mom**2 * (self.m1 + self.m2)/(G * self.m1**2 * self.m2**2)

        eccen_squared = 1 + 2*(self.m1 + self.m2)*self.energy*self.ang_mom**2/(G**2 * self.m1**3 * self.m2**3)
        eccen = np.sqrt(eccen_squared[0])   # conditions at the first time step should ideally determine the dynamics and hence the overall eccentricity 
        self.eccen = eccen

        # we need to measure phi_0 (phi at t=0) since phi = 0 happens when r = r_min (but we are not necessarily starting off at r_min)

        cos_phi_0 = rhs[0]/( self.eccen*np.linalg.norm(self.relative_r_vec[0]) ) - 1
        # phi_0 = np.arccos(cos_phi_0)
        
        # measure the cosine of the increment in phi at all timesteps (=delta_phi) as compared to the starting phi_0
        cos_delta_phi = np.sum(self.relative_r_vec[1:] * self.relative_r_vec[0], axis = 1)/(np.linalg.norm(self.relative_r_vec[1:], axis = 1) * np.linalg.norm(self.relative_r_vec[0]))
        
        sin_phi_0 = np.sqrt(1 - cos_phi_0**2)
        sin_delta_phi = np.sqrt(1 - cos_delta_phi**2)
        # FIXME - what happens after delta_phi becomes larger than pi?

        if np.linalg.norm(self.relative_r_vec[0]) < np.linalg.norm(self.relative_r_vec[1]):
            # r is increasing, phi_0 is positive 
            cos_phi = cos_phi_0 * cos_delta_phi - sin_phi_0 * sin_delta_phi
        else:
            # r is decreasing, phi=0 is yet to come, so phi_0 must be negative
            cos_phi = cos_phi_0 * cos_delta_phi + sin_phi_0 * sin_delta_phi

        # need to check the equality of these two across all timesteps - 

        assert np.allclose(np.linalg.norm(self.relative_r_vec) * (1 + self.eccen * cos_phi), rhs), "conic equation fails at some timestep"
        # both arguments of allclose here are Mx1 arrays
        # can pass a custom tolerance fraction as the input to allclose if needed

###############################################################


###############################################################

    def wrapper_for_analytical_check(self, tol_frac):

        self.transform_coords()
        self.compute_energy_mom(tol_frac = tol_frac)
        self.trajectory_checker()