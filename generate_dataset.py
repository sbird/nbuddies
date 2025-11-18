from src.ICs import *
from src.evolution import simulation
import os
import argparse
from pint import UnitRegistry 
import pickle as pkl

ureg = UnitRegistry()
ureg.define('solarmass = 1.98847e30 * kilogram') 
G = 6.67430e-11 * ureg.meter**3 / (ureg.kilogram * ureg.second**2) 
GG = G.to("kiloparsec * kilometer**2 / (solarmass * second**2)") 

# Create the parser
parser = argparse.ArgumentParser(prog="NBuddies mass dataset generation", description="Run many N-body simulations randomly distributed within a parameter space.", 
                                 epilog="Created in Fall 2025 quarter of PHYS 206: Computational Astrophysics at UCR.")

# Add arguments
# Required
parser.add_argument("N", type=int, help="Number of simulations to run")

parser.add_argument("Name", type=str, help="Name of your dataset")

parser.add_argument("--time", type=float, help="Time each should be allowed to run")

parser.add_argument("--seed", type = int, help = "Random seed for the initial conditions", default = 1)

parser.add_argument("--M_min", type = int, help = "Minimum BH Mass", default = 1e5)
parser.add_argument("--M_max", type = int, help = "Maximum BH Mass", default = 1e10)

parser.add_argument("--R_min", type = int, help = "Minimum Scale Radius", default = 1e-1)
parser.add_argument("--R_max", type = int, help = "Maximum Scale Radius", default = 1e1)

parser.add_argument("--N_min", type = int, help = "Minimum Number of BHs", default = 30)
parser.add_argument("--N_max", type = int, help = "Maximum Number of BHs", default = 100)

parser.add_argument("--clear", action=argparse.BooleanOptionalAction, 
                    help="Clears all previous data from this dataset", default=False)

# Parse the arguments
args = parser.parse_args()

nbuddies_path = os.path.dirname(os.path.realpath(__file__))
dataset_path = nbuddies_path+"/training_data"

#create directory if non-existent
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

if os.path.exists(dataset_path + "/" + args.Name + ".pkl") and not args.clear:
    with open(dataset_path + "/" + args.Name + ".pkl", 'rb') as f:
        dataset = pkl.load(f)
else:
    dataset = {"seeds" : np.asarray([]), "Ns" : np.asarray([]), "Ms" : np.asarray([]), "Rs" : np.asarray([])}

#prep params
[Ns, Ms, Rs] = np.random.rand(3,args.N)

Ns *= (args.N_max - args.N_min)
Ns += args.N_min
Ns = Ns.astype(int)

Ms *= (np.log(args.M_max) - np.log(args.M_min))
Ms += np.log(args.M_min)
Ms = np.exp(Ms)

Rs *= (np.log(args.R_max) - np.log(args.R_min))
Rs += np.log(args.R_min)
Rs = np.exp(Rs)

print(np.array(dataset["Ns"]))

for n in range(args.N):
    data_path = nbuddies_path+"/data/"+args.Name+f"_{len(dataset["seeds"])}"
    #create directory if non-existent
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    np.random.seed(args.seed)

    BHs, _ = generate_plummer_initial_conditions(
        n_blackholes=Ns[n],
        initial_mass=Ms[n],
        scale=Rs[n],
        ratio=0
    )
    print(
        f"Running {args.Name}_{n} with: N={Ns[n]}, R={Rs[n]}, M={Ms[n]}"
    )

    pkl.dump(BHs, open(data_path+"/ICs.pkl", "wb"))

    simulation(data_path+"/ICs.pkl", data_path, tot_time = args.time, nsteps = 20, delta_t = None,
           adaptive_dt = True, eta = 0.1, leapfrog = True, use_tree = True,
           use_dynamic_criterion = True, ALPHA = 0.1, THETA_0 = None)

    '''
    Harvest and save merger info
    '''

    #appends after each run so output is accurate even if stopped early
    dataset["seeds"] = np.append(dataset["seeds"], args.seed)
    dataset["Ns"] = np.append(dataset["Ns"], Ns[n])
    dataset["Ms"] = np.append(dataset["Ms"], Ms[n])
    dataset["Rs"] = np.append(dataset["Rs"], Rs[n])

    with open(dataset_path + "/" + args.Name + ".pkl", 'wb') as f:
        pkl.dump(dataset, f)


