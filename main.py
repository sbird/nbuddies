from src.ICs import *
from src.evolution import simulation
from src.visualizations import *
from src.Forces import GG, ALPHA, THETA_0
import os
import argparse
from pint import UnitRegistry 
from visualizations_comparisons import movie_3D_comparison
import pickle as pkl

ureg = UnitRegistry()
ureg.define('solarmass = 1.98847e30 * kilogram') 
G = 6.67430e-11 * ureg.meter**3 / (ureg.kilogram * ureg.second**2) 
GG = G.to("kiloparsec * kilometer**2 / (solarmass * second**2)") 

# Create the parser
parser = argparse.ArgumentParser(prog="NBuddies", description="Run N-body simulation on group of custom black holes.", 
                                 epilog="Created in Fall 2025 quarter of PHYS 206: Compuational Astrophysics at UCR.")

# Add arguments
# Required
parser.add_argument("N", type=int, help="Number of black holes in simulation")

parser.add_argument("R", type=float, help="Scale Plummer sphere radius (kpc) of black holes")

parser.add_argument("M", type=float, help="Mass (solar mass) of black holes")

# Optional
parser.add_argument("--name", type=str, help="Name of simulation run (default: mass_segregation)", default="mass_segregation")

parser.add_argument("--M_ratio", type=float, help="Set mass ratio between two types of black holes", default=0.0)

parser.add_argument("--n_steps", type=int, help="Number of steps for batch saving (default: 10)", default=10)

parser.add_argument("--adaptive_ts", action=argparse.BooleanOptionalAction, 
                    help="Use adaptive timestep formula (default: True)", default=True)

parser.add_argument("--eta", type=float, help="Set eta when using adaptive time step (default: 0.1)", default=0.1)

parser.add_argument("--use_tree", action=argparse.BooleanOptionalAction, 
                    help="Use Barnes Hut algorithm to calculate forces (default: True)", default=True)

parser.add_argument("--use_leapfrog", action=argparse.BooleanOptionalAction, 
                    help="Use Leap frog integration (default: True). False uses Euler integration", default=True)

parser.add_argument("--do_comp", action=argparse.BooleanOptionalAction,
                    help="Generate comparison movie between tree and brute force methods (default: True)", default=True)

parser.add_argument("--time", type = float, help="Total simulation time in seconds (default : 5e17)", default=5e17)

parser.add_argument("--IC_type", choices=["binary", "plummer"], default="plummer",
                    help="Choose type of initial condition: 'binary' or 'plummer' (default: plummer)")

# Parse the arguments
args = parser.parse_args()

R = args.R * ureg('kpc')
M = args.M * ureg('solarmass')

nbuddies_path = os.path.dirname(os.path.realpath(__file__))
data_path = nbuddies_path+"/data/"+args.name

# empty directory or create if non-existent
if not os.path.exists(data_path):
    os.makedirs(data_path)

sim_time = args.time * ureg('second')

print(f"sim time = {sim_time.to('Myr'):.3} = {sim_time.to('second'):.3}")

match args.IC_type:
    case "binary":
        if args.N != 2:
            raise ValueError(f"Binary initial conditions require N=2, but got N={args.N}")
        # Make binary ICs
        custom_vals = {
            'N': 2,
            'mass': np.array([1.0e7, 1.0e7]),
            'position': np.array([[1., 0., 0.], [-1., 0., 0.]]),
            'velocity': np.array([[0., 3.2791, 0.], [0., -3.2791, 0.]])
        }
        BHs, _ = generate_binary_ICs(N_BH=2, custom_vals=custom_vals)
        print(
            f"Running {args.name} with: N={custom_vals['N']}, M={custom_vals['mass']}, "
            f"n_steps={args.n_steps}, adaptive_ts={args.adaptive_ts}, eta={args.eta}, "
            f"use_leapfrog={args.use_leapfrog}"
        )

    case "plummer":
        BHs, _ = generate_plummer_initial_conditions(
            n_blackholes=args.N,
            initial_mass=(M / args.N).magnitude,
            scale=R.magnitude,
            ratio=args.M_ratio
        )
        print(
            f"Running {args.name} with: N={args.N}, R={args.R}, M={args.M}, "
            f"M_ratio={args.M_ratio}, n_steps={args.n_steps}, adaptive_ts={args.adaptive_ts}, "
            f"eta={args.eta}, use_tree={args.use_tree}, use_leapfrog={args.use_leapfrog}"
        )

    case _:
        raise ValueError(f"Invalid initial condition type: {args.IC_type}")

pkl.dump(BHs, open(data_path+"/ICs.pkl", "wb"))


# #calc relax time
# t_relax = 0.14*args.N * (R**(3/2)) / (np.log(0.4*args.N) * np.sqrt(GG*M))
# print(t_relax.to("Myr"))

# t_segregate = 0.1 * t_relax
# sim_time = 3*t_relax

# print(f"sim time = {sim_time.to('Myr'):.3} = {sim_time.to('second'):.3}")

#run_sim
# if args.do_comp:
#     simulation(data_path+"/ICs.pkl", data_path+"_tree", tot_time=sim_time.to('second').magnitude, nsteps=args.n_steps, 
#         adaptive_dt=args.adaptive_ts, eta=args.eta, use_tree=True)
#     movie_3D(args.name+"_tree")
#     simulation(data_path+"/ICs.pkl", data_path+"_brute", tot_time=sim_time.to('second').magnitude, nsteps=args.n_steps, 
#         adaptive_dt=args.adaptive_ts, eta=args.eta, use_tree=False)
#     movie_3D(args.name+"_brute")
#     movie_3D_comparison(args.name)
# else:
#     simulation(data_path+"/ICs.pkl", data_path, tot_time=sim_time.to('second').magnitude, nsteps=args.n_steps, 
#            adaptive_dt=args.adaptive_ts, eta=args.eta, use_tree=args.use_tree)
#     movie_3D(args.name)
    # radial_position_plot(args.name)

simulation(data_path+"/ICs.pkl", data_path, tot_time=sim_time.to('second').magnitude, nsteps=args.n_steps, 
           adaptive_dt=args.adaptive_ts, eta=args.eta, use_tree=args.use_tree)
movie_3D(args.name)
 # radial_position_plot(args.name)

 
################################ FIX BELOW ################################
alpha = ALPHA # Set tree parameter in dynamic criterion
theta = THETA_0 # Set threshold parameter in Barnes Hut algorithm 