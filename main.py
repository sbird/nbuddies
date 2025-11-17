from src.ICs import *
from src.evolution import simulation
from src.visualizations import *
from src.Forces import GG
import os
import argparse
from pint import UnitRegistry 
from visualizations_comparisons import movie_3D_comparison
import pickle as pkl
import time

ureg = UnitRegistry()
ureg.define('solarmass = 1.98847e30 * kilogram') 
G = 6.67430e-11 * ureg.meter**3 / (ureg.kilogram * ureg.second**2) 
GG = G.to("kiloparsec * kilometer**2 / (solarmass * second**2)") 

# Create the parser
parser = argparse.ArgumentParser(prog="NBuddies", description="Run N-body simulation on group of custom black holes.", 
                                 epilog="Created in Fall 2025 quarter of PHYS 206: Computational Astrophysics at UCR.")

# Add arguments
# Required
parser.add_argument("N", type=int, help="Number of black holes in simulation")

parser.add_argument("R", type=float, help="Scale Plummer sphere radius (kpc) of black holes")

parser.add_argument("M", type=float, help="Mass (solar mass) of black holes")

# Optional
parser.add_argument("--name", type=str, help="Name of simulation run (default: mass_segregation)", default="mass_segregation")

parser.add_argument("--M_ratio", type=float, help="Set mass ratio between two types of black holes", default=0.0)

parser.add_argument("--n_steps", type=int, help="Number of steps for batch saving (default: 10)", default=10)

parser.add_argument("--fixed_ts", action=argparse.BooleanOptionalAction, 
                    help="Use fixed timestep", default=False)

parser.add_argument("--delta_t", type=float, help="Set timestep if using fixed timestep", default=None)

parser.add_argument("--eta", type=float, help="Set eta when using adaptive time step (default: 0.1)", default=0.1)

parser.add_argument("--brute_force", action=argparse.BooleanOptionalAction, 
                    help="Do brute force O(N^2) force calculation", default=False)

parser.add_argument("--use_geometric_criterion", action=argparse.BooleanOptionalAction, 
                    help="Use geometric criterion to determine if nodes of Barnes Hut tree can be approximated as point masses. If false uses dynamic criterion (default: False)", default=False)

parser.add_argument("--ALPHA", type=float, help="Set accuracy parameter for dynamic node approximation criterion (default: 0.1)", default=0.1)

parser.add_argument("--THETA_0", type=float, help="Set accuracy parameter for geometric node approximation criterion (default: 0.1)", default=0.1)

parser.add_argument("--use_euler", action=argparse.BooleanOptionalAction, 
                    help="Use Euler integration. Default behavior uses Euler integration", default=False)

# parser.add_argument("--time", type = float, help="Total simulation time in seconds (default : 5e17)", default=5e17)

parser.add_argument('--x_time', type=int, help="Time scaling factor (default: 3)", default=3)

parser.add_argument("--IC_type", choices=["binary", "plummer"], default="plummer",
                    help="Choose type of initial condition: 'binary' or 'plummer' (default: plummer)")

# Parse the arguments
args = parser.parse_args()

start = time.time()

if args.delta_t is None and args.fixed_ts:
    raise("Timestep must be specified if not using adaptive timestep")

R = args.R * ureg('kpc')
M = args.M * ureg('solarmass')

nbuddies_path = os.path.dirname(os.path.realpath(__file__))
data_path = nbuddies_path+"/data/"+args.name

#create directory if non-existent
if not os.path.exists(data_path):
    os.makedirs(data_path)

if args.N > 2:
    #calc relax time
    t_relax = 0.14*args.N * (R**(3/2)) / (np.log(0.4*args.N) * np.sqrt(GG*M))
    print(t_relax.to("Myr"))
    # t_segregate = 0.1 * t_relax
    sim_time = args.x_time * t_relax
    # sim_time = args.time * ureg('second')
else:
    t_orbit = 2 * np.pi * np.sqrt((R**3) / (GG * M))
    sim_time = args.x_time * t_orbit

print(f"sim time = {sim_time.to('Myr'):.3} = {sim_time.to('second'):.3}")

match args.IC_type:
    case "binary":
        if args.N != 2:
            raise ValueError(f"Binary initial conditions require N=2, but got N={args.N}")
        # Make binary ICs
        vel = np.sqrt( GG * M / (4 * R)  ) / (ureg.km / ureg.s)
        custom_vals = {
            'N': args.N,
            'mass': np.array([args.M, args.M]),
            'position': np.array([[args.R, 0., 0.], [-args.R, 0., 0.]]),
            'velocity': np.array([[0., vel, 0.], [0., -vel, 0.]])
        }
        BHs, _ = generate_binary_ICs(N_BH=2, custom_vals=custom_vals)

        print(
            f"Running {args.name} with: N={args.N}, M={M}, "
            f"n_steps={args.n_steps}, adaptive_ts={not args.fixed_ts}, eta={args.eta}, "
            f"use_leapfrog={not args.use_euler}"
        )

    case "plummer":
        BHs, _ = generate_plummer_initial_conditions(
            n_blackholes=args.N,
            initial_mass=(M).magnitude,
            scale=R.magnitude,
            ratio=args.M_ratio
        )
        print(
            f"Running {args.name} with: N={args.N}, R={args.R}, M={args.M}, "
            f"M_ratio={args.M_ratio}, n_steps={args.n_steps}, adaptive_ts={not args.fixed_ts}, "
            f"eta={args.eta}, use_tree={not args.brute_force}, use_leapfrog={not args.use_euler}"
        )

    case _:
        raise ValueError(f"Invalid initial condition type: {args.IC_type}")

pkl.dump(BHs, open(data_path+"/ICs.pkl", "wb"))



# print(f"sim time = {sim_time.to('Myr'):.3} = {sim_time.to('second'):.3}")

simulation(data_path+"/ICs.pkl", data_path, tot_time=sim_time.to('second').magnitude, nsteps=args.n_steps, delta_t = args.delta_t,
           adaptive_dt= not args.fixed_ts, eta=args.eta, leapfrog= not args.use_euler, use_tree=not args.brute_force,
           use_dynamic_criterion= not args.use_geometric_criterion, ALPHA = args.ALPHA, THETA_0 = args.THETA_0)

movie_3D(args.name)
radial_position_plot(args.name)
end = time.time()
print(f"Total execution time: {((end - start)/60):.2f} minutes")