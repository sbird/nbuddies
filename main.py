from src.ICs import *
from src.evolution import simulation
from src.visualizations import *
from src.Forces import GG
import os
from pint import UnitRegistry 

ureg = UnitRegistry()
ureg.define('solarmass = 1.98847e30 * kilogram') 
G = 6.67430e-11 * ureg.meter**3 / (ureg.kilogram * ureg.second**2) 
GG = G.to("kiloparsec * kilometer**2 / (solarmass * second**2)") 

#sim parameters
name = "mass_segregation"
N = 50
R = 5 * ureg('kpc')
M = 5e8 * ureg('solarmass')

nbuddies_path = os.path.dirname(os.path.realpath(__file__))
data_path = nbuddies_path+"/data/"+name

# empty directory or creat if not exist
if not os.path.exists(data_path):
    os.makedirs(data_path)

# Make ICs
BHs, masses = generate_plummer_initial_conditions(n_blackholes=N, initial_mass=(M/N).magnitude, scale=R.magnitude, ratio=0.1)

pkl.dump(BHs, open(data_path+"/ICs.pkl", "wb"))

#calc relax time
t_relax = 0.14*N * (R**(3/2)) / (np.log(0.4*N) * np.sqrt(GG*M))
print(t_relax.to("Myr"))

t_segregate = 0.1 * t_relax

sim_time = (2*t_segregate).to("second").magnitude

print(f"sim time = {2*t_segregate.to('Myr'):.3} = {sim_time:.3} sec")
#run_sim
simulation(data_path+"/ICs.pkl", data_path, tot_time=sim_time, nsteps=10, adaptive_dt=True, eta=0.1, use_tree=True)

#visualize
movie_3D(name)
radial_position_plot(name)