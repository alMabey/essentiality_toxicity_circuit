import sys
import os
import argparse
from pathlib import Path
import multiprocessing as mp
import time
import numpy as np
# the parent directory
directory = os.path.abspath(os.path.join(".."))
sys.path.append(directory)

from scripts.model_params import *              # noqa: E402
from scripts.cell_model_batch_culture import *  # noqa: E402
from scripts.batch_culture_simulation import *  # noqa: E402

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.core.callback import Callback
from pymoo.core.sampling import Sampling
from pymoo.factory import get_sampling

class PopulationSampling(Sampling):
    def __init__(self, X):
        super().__init__()
        self.X = X

    def _do(self, problem, n_samples, random_state=None):
        return self.X


# all of the names of the params in xPR, so that given an index
# the code knows which parameter was optimised
PARAM_NAMES = [
    "w0", "wT", "wE", "wEp", "wTF", "wpTox", "wTp", "k_Ep", "Km_Ep",
    "k_Tp", "Km_Tp", "a_energy_pTox", "a_elongation_pTox",
    "K_E", "K_pTox", "kdiffP", "VolCell", "VolCult", "ksf", "ksr"
]

# Functions to convert the strings from the command line arguments into python objects
def parse_int_list(s):
    return [int(x.strip()) for x in s.split(",") if x.strip() != ""]

def parse_float_list(s):
    return [float(x.strip()) for x in s.split(",") if x.strip() != ""]

def parse_topology(s):
    """
    Parse topology string: comma-separated ints, e.g. "1,0,0,1,1,0"
    """
    vals = [int(x.strip()) for x in s.split(",") if x.strip() != ""]
    return vals

# A function which builds the function that simulates for a given 
# set of optimisation variables and calculates the volumetric
# productivity and yield
def make_prod_yield_func(base_params, hPR, xPR, param_indices):

    base_params = list(base_params)
    hPR = np.array(hPR, copy=True)
    xPR_base = np.array(xPR, copy=True)
    param_indices = list(param_indices)

    def prod_yield_calc(x_vector):
        #local copy of xPR
        xPR_local = xPR_base.copy()
        #substituting in the optimisation parameters
        for idx, value in zip(param_indices, x_vector):
            xPR_local[idx] = value
        #calculating the volumetric productivity and yield
        #for this calculation
        vP, pY = batch_cult_prod_yield_calc(base_params, hPR, xPR_local)
        return -np.log(vP), -np.log(pY)

    return prod_yield_calc


# Setting up problem definitions for pymoo; 
# SingleObjective is for calculating the absolute maximum vP/pY, 
# which are used for scaling in the multiobjectivity optimisation
# MultiObjectiveScaled does the multiobjective optimisaiton
class SingleObjective(ElementwiseProblem):
    def __init__(self, calculate_prod_yield_func, obj_id, n_params, lower_bounds,
                 upper_bounds, parallelization=None):
        super().__init__(
            n_var=n_params,
            n_obj=1,
            xl=np.array(lower_bounds),
            xu=np.array(upper_bounds),
            parallelization=parallelization,
        )
        self.calculate_prod_yield_func = calculate_prod_yield_func
        self.obj_id = obj_id

    def _evaluate(self, x, out):
        obj_vec = self.calculate_prod_yield_func(x)
        out["F"] = obj_vec[self.obj_id]


class MultiObjectiveScaled(ElementwiseProblem):
    def __init__(self, calculate_prod_yield_func, n_params, lower_bounds, upper_bounds,
                 scale, parallelization=None):
        super().__init__(
            n_var=n_params,
            n_obj=2,
            xl=np.array(lower_bounds),
            xu=np.array(upper_bounds),
            parallelization=parallelization,
        )
        self.calculate_prod_yield_func = calculate_prod_yield_func
        self.scale = np.array(scale, dtype=float)

    def _evaluate(self, x, out):
        vP_log, pY_log = self.calculate_prod_yield_func(x)
        out["F"] = np.array([vP_log, pY_log]) / self.scale

# Callback for on-the-fly Pareto front checkpointing
# every N generations, write to disk the 
# current pareto front, the corresponding vP/pY, 
# the parameter indices being optimised, 
# a full list of parameter names, etc
class ParetoCheckpointCallback(Callback):
    def __init__(self, partial_path, param_indices, save_every=10):
        super().__init__()
        self.partial_path = Path(partial_path)
        self.param_indices = list(param_indices)
        self.save_every = save_every

    def notify(self, algorithm):
        #check if this is a checkpoint generation where the current pareto
        #front estimate is saved
        gen = algorithm.n_gen
        if gen % self.save_every != 0:
            return

        #extract the current estimated pareto front
        opt = algorithm.opt  # non-dominated set
        if opt is None or len(opt) == 0:
            return

        #get the parameter sets and vP/pY for the current
        #estimated pareto front
        pareto_X = opt.get("X")
        pareto_F = opt.get("F")

        #storing the names of the parameters which are optimised
        varied_param_names = [PARAM_NAMES[i] for i in self.param_indices]

        #saving the current pareto front estimate
        np.savez_compressed(
            self.partial_path,
            pareto_X=pareto_X,
            pareto_F=pareto_F,
            param_indices=np.array(self.param_indices, dtype=int),
            param_names=np.array(PARAM_NAMES),
            varied_param_names=np.array(varied_param_names),
            n_gen=int(gen),
            timestamp=time.time(),
        )
        #printing confirmation of the checkpointing
        print(f"[Checkpoint] Saved partial Pareto front at generation {gen} "
              f"to {self.partial_path.name}", flush=True)


# the main optimisation routine
def main():
    #########################################################################################
    #firstly defining the parser, which takes all of the command line arguments and 
    #converts them to python variables which can be used in the optimisation pipeline

    #defining the parser object
    parser = argparse.ArgumentParser(
        description="HPC NSGA-II optimisation for batch culture model."
    )
    #########################################################################################
    #now adding all of the possible arguments for the simulation, e.g. the name of the 
    #current optimisation case, the particular parameters we are optimising over, 
    #the bounds, the export mechanism, the GA hyperparameters etc
    parser.add_argument(
        "--case",
        type=str,
        required=True,
        help="Name of current optimisation case"
    )
    parser.add_argument(
        "--param-indices",
        type=str,
        required=True,
        help=("Comma-separated indices of xPR to optimise, "
              "e.g. '2,3,14'.")
    )
    parser.add_argument(
        "--lower-bounds",
        type=str,
        required=True,
        help="Comma-separated lower bounds for each optimised parameter."
    )
    parser.add_argument(
        "--upper-bounds",
        type=str,
        required=True,
        help="Comma-separated upper bounds for each optimised parameter."
    )
    parser.add_argument(
        "--topology",
        type=str,
        required=True,
        help=("Comma-separated topology flags "
              "(lin_trans,Tp_trans,T_trans,diff_trans,eprodtox,elongationtox), "
              "e.g. '1,0,0,0,1,0'.")
    )
    parser.add_argument(
        "--pop-size-ga",
        type=int,
        default=30,
        help="Population size for single-objective runs"
    )
    parser.add_argument(
        "--pop-size-nsga2",
        type=int,
        default=60,
        help="Population size for NSGA-II multi-objective run"
    )
    parser.add_argument(
        "--n-gen",
        type=int,
        default=80,
        help="Number of generations for NSGA-II."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Seed for all optimisation runs."
    )
    #Number of worker processes for parallelisation
    parser.add_argument(
        "--n-proc",
        type=int,
        default=None,
        help=("Number of worker processes for parallel evaluation. "
              "Defaults to $SLURM_CPUS_PER_TASK or all available cores.")
    )
    #Checkpointing arguments
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help=("Checkpoint Pareto front every N generations "
              "(NSGA-II only).")
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output_data",
        help="Directory (relative to repo root) to store NPZ outputs."
    )
    parser.add_argument("--pop-size-ga", type=int, default=30,
                    help="Population size for the single-objective GAs")

    parser.add_argument("--n-gen-ga", type=int, default=100,
                        help="Number of generations for the single-objective GAs")

    parser.add_argument("--pop-size-nsga2", type=int, default=60,
                        help="Population size for NSGA-II multi-objective optimisation")

    parser.add_argument("--n-gen-nsga2", type=int, default=80,
                        help="Number of generations for NSGA-II")

    parser.add_argument("--mutation-prob", type=float, default=None,
                        help="Mutation probability for NSGA-II (default = 1 / n_params)")

    parser.add_argument("--crossover-eta", type=float, default=15.0,
                        help="SBX crossover eta value")

    parser.add_argument("--mutation-eta", type=float, default=20.0,
                        help="Polynomial mutation eta value")

    #########################################################################################
    #parsing the command line arguments into python objects
    args = parser.parse_args()
    param_indices = parse_int_list(args.param_indices)
    lower_bounds = parse_float_list(args.lower_bounds)
    upper_bounds = parse_float_list(args.upper_bounds)
    topology = np.array(parse_topology(args.topology), dtype=int)

    if not (len(param_indices) == len(lower_bounds) == len(upper_bounds)):
        raise ValueError(
            "param_indices, lower_bounds, and upper_bounds must all be the "
            "same length."
        )

    n_params = len(param_indices)

    #########################################################################################
    #setting up the multiprocessing parallelisation
    #if a command line argument was given take it, 
    #otherwise just take the SLURM_CPUS_PER_TASK batch argument
    if args.n_proc is not None:
        nproc = args.n_proc
    else:
        nproc = int(os.getenv("SLURM_CPUS_PER_TASK", mp.cpu_count()))
    print(f"Using {nproc} processes for parallel evaluation.", flush=True)

    #########################################################################################
    #Setting up the base simulation parameter (base_params) and model parameter (hPR, xPR) arrays 
    sS0 = 0.5           # quality of nutrients
    cultvol = 1.25      # culture volume
    vX = 726
    kX = 1e3

    tmax = 20000
    runintmax = 1e6

    N0 = 1e6
    xS0 = 4.180555555555556e22  # 10 g/L glucose in 1.25 L
    M0 = 1e8

    base_params = [xS0, runintmax, tmax, N0, topology]

    #Create parameter arrays
    hPR0, xPR0 = model_params(sS0, vX, kX, cultvol, leaky_control=False)
    hPR = np.array(hPR0)
    xPR = np.array(xPR0)

    #the names of the parameters that we are optimising over
    varied_param_names = [PARAM_NAMES[i] for i in param_indices]
    #now printing the exact parameters we are optimising and their bounds
    print("Optimising over parameters:")
    for idx, name, lb, ub in zip(param_indices, varied_param_names,
                                 lower_bounds, upper_bounds):
        print(f"  index {idx:2d} -> {name:>15s}, bounds = [{lb}, {ub}]")
    #########################################################################################
    #Setting up the code output paths
    repo_root = Path(directory)
    out_dir = repo_root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    final_file = out_dir / f"{args.case}_pareto_results.npz"
    partial_file = out_dir / f"{args.case}_pareto_partial.npz"

    if final_file.exists():
        print(f"Final result file already exists: {final_file}")
        print("Nothing to do. Delete it if you want to recompute.")
        return

    #Constructing the pY/vP evaluation function
    calculate_prod_yield_func = make_prod_yield_func(base_params, hPR, xPR, param_indices)

    #Parallel pool for pymoo
    with mp.Pool(processes=nproc) as pool:

        #starmultiprocessing pattern, the best for pymoo
        parallelization = ("starmap", pool.starmap)
        #
        #Running the single-objective runs to calculate the normalisation scale
        print("\nMaximising vP...", flush=True)
        problem_vP = SingleObjective(
            calculate_prod_yield_func=calculate_prod_yield_func,
            obj_id=0,
            n_params=n_params,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            parallelization=parallelization,
        )

        res_vP = minimize(
            problem_vP,
            GA(pop_size=args.pop_size_ga, n_gen=args.n_gen_ga),
            seed=args.seed,
            verbose=True,
        )
        max_vP_log = res_vP.F[0]

        print("\nMaximising pY...", flush=True)
        problem_pY = SingleObjective(
            calculate_prod_yield_func=calculate_prod_yield_func,
            obj_id=1,
            n_params=n_params,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            parallelization=parallelization,
        )

        res_pY = minimize(
            problem_pY,
            GA(pop_size=args.pop_size_ga, n_gen=args.n_gen_ga),
            seed=args.seed,
            verbose=True,
        )
        max_pY_log = res_pY.F[0]

        scale = np.array([abs(max_vP_log), abs(max_pY_log)], dtype=float)
        print(f"\nscale = {scale}\n", flush=True)

        #Now running the full multiobjective optimisation
        print("Starting multi-objective optimisation...", flush=True)
        #setting up the problem
        problem_moo = MultiObjectiveScaled(
            calculate_prod_yield_func=calculate_prod_yield_func,
            n_params=n_params,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            scale=scale,
            parallelization=parallelization,
        )

        #checking if a partial checkpoint exists, use it to seed the initial population
        sampling_array = None

        if partial_file.exists():
            try:
                data = np.load(partial_file, allow_pickle=True)
                pareto_X_partial = np.array(data["pareto_X"])
                n_partial = len(pareto_X_partial)

                pop_size = args.pop_size_nsga2
                n_params = len(param_indices)

                if n_partial == pop_size:
                    # Use checkpoint individuals directly
                    sampling_array = pareto_X_partial
                    print(f"Restarting NSGA-II with {pop_size} individuals from checkpoint.")
                else:
                    raise ValueError("Checkpoint size != population size; deterministic restart requires equality.")

            except Exception as e:
                print(f"Warning loading checkpoint: {e}")
                sampling_array = None

        # Convert to proper Sampling object
        if sampling_array is not None:
            sampling = PopulationSampling(sampling_array)
        else:
            sampling = get_sampling("real_random")


        algorithm = NSGA2(
            pop_size=args.pop_size_nsga2,
            sampling=sampling,
            eliminate_duplicates=True,
        )

        termination = get_termination("n_gen", args.n_gen)
        callback = ParetoCheckpointCallback(
            partial_path=partial_file,
            param_indices=param_indices,
            save_every=args.checkpoint_every,
        )

        start = time.time()
        res = minimize(
            problem_moo,
            algorithm,
            ("n_gen", args.n_gen_nsga2),
            termination,
            seed=args.seed,
            verbose=True,
            callback=callback,
        )
        elapsed = time.time() - start
        print(f"Multiobjective optimisation completed in {elapsed:.1f} s", flush=True)

    #postprocessing and final .npz output
    #taking the optimal parameters+unscaling the optimal vPs, pYs
    pareto_X = res.X                     
    pareto_scaled = res.F             
    pareto_unscaled = pareto_scaled * scale 

    vP_vals = np.exp(-pareto_unscaled[:, 0])
    pY_vals = np.exp(-pareto_unscaled[:, 1])

    varied_param_names = [PARAM_NAMES[i] for i in param_indices]

    #Now saving to a .npz file
    np.savez_compressed(
        final_file,
        pareto_X=pareto_X,
        pareto_F=pareto_unscaled,
        vP=vP_vals,
        pY=pY_vals,
        param_indices=np.array(param_indices, dtype=int),
        param_names=np.array(PARAM_NAMES),
        varied_param_names=np.array(varied_param_names),
        topology=np.array(topology),
        scale=scale,
        seed=args.seed,
        n_gen=args.n_gen,
    )

    print(f"\nSaved final Pareto front to: {final_file.name}")
    if partial_file.exists():
        print(f"Partial checkpoint retained at: {partial_file.name}")


if __name__ == "__main__":
    main()
