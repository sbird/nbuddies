from BlackHoles_Struct import BlackHole
import numpy as np 
from pint import UnitRegistry
import pickle
import os

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
    path: the path to the directory where the analytical check results will be saved.
    output:
    TBD exactly 
    """
    
    def __init__(self, path):

        self.path = path
        ## list all the files in the directory and list the index of the batches
        files = os.listdir(path)
        batch_list = [int(i.split('h')[-1].split('.pkl')[0]) for i in files]
        # batch_list.sort()
        # self.tot_times = len(batch_list)  # total number of time steps including initial
        self.tot_times = batch_list[0]  # total number of time steps including initial

        ## Store the data from all batches in an array
        self.positions = np.array( self.tot_times * [np.zeros((2,3))] ) # shape = (num_batches, 2, 3)
        self.velocities = np.array( self.tot_times * [np.zeros((2,3))] ) # shape = (num_batches, 2, 3)
        # Read the data from each batch and store in the above arrays
        # for batch in batch_list:
        with open(f'{path}/data_batch{self.tot_times}.pkl', 'rb') as f:
            BH_data_final = pickle.load(f)['data']

        if np.shape(BH_data_final)[-1] != 2:
            raise ValueError("Analytical solution only exists for the two-body problem. Limit number of black holes to 2.")

        for batch in range(self.tot_times):
            for i in range(2):
                self.positions[batch][i] = BH_data_final[-1][i].position # position of ith BH at this batch
                self.velocities[batch][i] = BH_data_final[-1][i].velocity # velocity of ith BH at this batch

                self.m1 = BH_data_final[0][0].mass * mass_unit# mass of BH 1
                self.m2 = BH_data_final[0][1].mass * mass_unit # mass of BH 2
        # assuming the time step stacking to be done such that the last dimension is the number of total timesteps 
        self.transform_coords()

#######################################################


#######################################################

    def transform_coords(self):
        """
        transforms the exisiting position and velocity coordinates into a single, relative coordinate
        which appears in the solution to the 2-body problem alongside the reduced mass. 
        """

        r1_vec = self.positions[:,0,:] * dist_unit # shape = (num_batches, 3)
        r2_vec = self.positions[:,1,:] * dist_unit # shape = (num_batches, 3)
        # both of these will be M x 3 arrays (with the 3rd column being always zero ideally)
        # hereby M is the total number of timesteps (including initial)

        # assert r1_vec[:, 2] == 0 and r2_vec[:, 2] == 0, "WARNING: motion of the two bodies deviates from planarity."

        # FIXME - if planar, can do away with the third column 

        v1_vec = self.velocities[:,0,:] * vel_unit # shape = (num_batches, 3)
        v2_vec = self.velocities[:,1,:] * vel_unit # shape = (num_batches, 3)

        # as mentioned in the docstring, no need for dimensional consistency checks on these
        # remember that relative_r_vec is the same in any frame (CM or not)
        self.COM_r = (self.m1 * r1_vec + self.m2 * r2_vec) / (self.m1 + self.m2)
        self.COM_v = (self.m1 * v1_vec + self.m2 * v2_vec) / (self.m1 + self.m2)
        self.relative_v_vec = v1_vec - v2_vec  # M x 3 array
        self.relative_r_vec = r1_vec - r2_vec  # M x 3 array
        self.mu = self.m1*self.m2/(self.m1 + self.m2)  # scalar

############################################################
        

############################################################

    def compute_energy_mom(self, tol_frac = 1e-3):
        print("The tolerance fraction for checking conservation is set to %.1f %%" % (tol_frac*100))
        """computes two arrays that give the ang_mom and energy at each timestep, can readily check for conservation
       
       inputs: 
       tol_frac: sets the tolerance fraction for checking the constancy of energy and momentum    
        """
        ang_mom = self.mu * np.linalg.norm(np.cross(self.relative_r_vec, self.relative_v_vec), axis = 1) # axis = 1 implies that self.relative_r_vec and v are Mx3 arrays and not 3xM
        # follows from L = r cross p for the effective 1 body coordinates
        # M x 1 array
        potential = -(G*self.m1*self.m2/(np.linalg.norm(self.relative_r_vec, axis = 1))).to('kg*km**2/s**2')
        energy = 1/2*(self.m1+self.m2)*np.sum(self.COM_v**2, axis = 1) + ang_mom**2/(2*self.mu*np.sum(self.relative_r_vec**2, axis = 1)) + potential
        # follows from E = 1/2 (m_1+m_2) V^2 + L^2/(2 mu r^2) - G m_1 m_2/r for the effective 1 body coordinates
        # M x 1 array

        assert np.shape(ang_mom)[0] == self.tot_times and np.shape(energy)[0] == self.tot_times, "dimensions of energy and ang_momentum data do not match the total number of timesteps"

        # Now check for conservation of both
        if np.all(ang_mom - ang_mom[0] < tol_frac * ang_mom[0]):
            print(f"Angular momentum is conserved at the {np.max(ang_mom/ang_mom[0] - 1)*100:.4f} % level from the starting value")
        else:
            print(f"Angular momentum is NOT conserved at the {tol_frac*100} % level from the starting value")

        if np.all((energy - energy[0]) / energy[0] < tol_frac):
            print(f"Energy is conserved at the {np.max(energy/energy[0] - 1)*100:.4f} % level from the starting value")
        else:
            print(f"Energy is NOT conserved at the {tol_frac*100} % level from the starting value")
        # seems like a better option because more explicitly states the level it is conserved at? 

        # assert np.all(ang_mom - ang_mom[0] < tol_frac * ang_mom[0]), f"Angular momentum is not conserved at the {tol_frac*100} % level from the starting value"
        assert np.all((energy - energy[0]) / energy[0] < tol_frac), f"Energy is not conserved at the {tol_frac*100} % level from the starting value"

        self.ang_mom = ang_mom
        self.energy = energy

        return ang_mom, energy

    ############################################################


    ############################################################

    def trajectory_checker(self):
        """
        makes the more stringent test of checking if the relative coordinates satisfy the general conic equation
        """

        # evaluate rhs (RHS of the equation r(1+cos(phi)) = rhs as described in the comment above this class) 
        """
        rhs = L**2 / (mu**2 * G * (m1 + m2))
        e = sqrt( 1 + 2 * epsilon * h**2 / sgp**2 )
        h = angular momentum / mu       [Specific Angular Momentum]
        epsilon = Total Energy / mu     [Specific Energy]
        sgp = G * (m1 + m2)             [Standard Gravitational Parameter]
        """

        rhs = self.ang_mom**2 * (self.m1 + self.m2)/(G * self.m1**2 * self.m2**2)
        eccen_squared = 1 + 2*(self.m1 + self.m2)*self.energy*self.ang_mom**2/(G**2 * self.m1**3 * self.m2**3)
        eccen = np.sqrt(eccen_squared[0])   # conditions at the first time step should ideally determine the dynamics and hence the overall eccentricity 
        self.eccen = eccen

        # we need to measure phi_0 (phi at t=0) since phi = 0 happens when r = r_min (but we are not necessarily starting off at r_min)
        # print(rhs, np.linalg.norm(self.relative_r_vec[0]))
        ref_r = np.linalg.norm(self.relative_r_vec[0])
        cos_phi_0 = (rhs[0]/ref_r - 1 ) / self.eccen
        assert np.abs(cos_phi_0) <= 1, "cos(phi_0) is not physical, check the initial conditions"
        # phi_0 = np.arccos(cos_phi_0)
        # measure the cosine of the increment in phi at all timesteps (=delta_phi) as compared to the starting phi_0
        cos_delta_phi = np.sum(self.relative_r_vec[1:] * ref_r, axis = 1)/(np.linalg.norm(self.relative_r_vec[1:], axis = 1) * np.linalg.norm(ref_r))
        
        sin_phi_0 = np.sqrt(1 - cos_phi_0**2)
        sin_delta_phi = np.sqrt(1 - cos_delta_phi**2)
        assert np.abs(sin_phi_0) <= 1, "cos(phi_0) is not physical, check the initial conditions"
        # FIXME - what happens after delta_phi becomes larger than pi?

        # print(cos_phi_0 * cos_delta_phi - sin_phi_0 * sin_delta_phi)
        if np.linalg.norm(self.relative_r_vec[0]) < np.linalg.norm(self.relative_r_vec[1]):
            # r is increasing, phi_0 is positive 
            cos_phi = cos_phi_0 * cos_delta_phi - sin_phi_0 * sin_delta_phi
        else:
            # r is decreasing, phi=0 is yet to come, so phi_0 must be negative
            cos_phi = cos_phi_0 * cos_delta_phi + sin_phi_0 * sin_delta_phi

        # need to check the equality of these two across all timesteps - 
        # print(np.linalg.norm(self.relative_r_vec) * (1 + self.eccen * cos_phi), rhs[1:])
        assert np.allclose(np.linalg.norm(self.relative_r_vec) * (1 + self.eccen * cos_phi), rhs[1:]), "conic equation fails at some timestep"
        # both arguments of allclose here are Mx1 arrays
        # can pass a custom tolerance fraction as the input to allclose if needed

###############################################################



# output_dir = "./data"
# check1 = AnalyticalCheck(output_dir)
# check1.wrapper_for_analytical_check(tol_frac = 5e-2)