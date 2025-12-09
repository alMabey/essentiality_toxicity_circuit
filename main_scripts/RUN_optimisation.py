import os
import multiprocessing as mp
import numpy as np
import pandas as pd
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination

# importing simulation functions
from scripts.model_params import *
from scripts.cell_model import *
from scripts.batch_culture_simulation import *

####################################################################################################################
#The optimisation case
case_name = "batch_linear_energy"

#Setting the multiobjective optimisation hyperparameters
n_gens_single = 50
pop_size_single = 30

n_gens_multi = 80
pop_size_multi = 60

#the precise model
lin_trans = 1
Tp_trans = 0
T_trans = 0
diff_trans = 0

eprodtox = 1
elongationtox = 0

topology = [lin_trans, Tp_trans, T_trans, diff_trans, eprodtox, elongationtox]
####################################################################################################################
#The base parameters

sS0 = 0.5; cultvol = 1.25
vX = 726; kX = 1e3
tmax = 20000; runintmax = 1e6

N0  = 1e6
xS0 = 4.180555555555556e+22
M0  = 1e8

base_params = [xS0, 0, runintmax, tmax, N0, topology]

hPR0, xPR0 = model_params(sS0, vX, kX, cultvol, leaky_control=False)
hPR = np.array(hPR0)
xPR = np.array(xPR0)

##############################################################################################
# setting the parameter indices relative to xPR and their bounds

all_param_labels = [
    "w0", "wT", "wE", "wEp", "wTF", "wpTox", "wTp",
    "k_Ep", "Km_Ep",
    "k_Tp", "Km_Tp",
    "a_energy_pTox", "a_elongation_pTox",
    "K_E", "K_pTox", "kdiffP",
    "VolCell", "VolCult",
    "ksf", "ksr"
]

param_indices = [1, 2, 3, 4, 5, 13, 14]             # e.g., wEp at index 2 (old behaviour)
optimised_labels = [all_param_labels[i] for i in param_indices]
lower_bounds  = [0, 0, 0, 0, 0, 0, 0]             # lower bounds for each parameter
upper_bounds  = [200, 200, 200, 200, 200, 200, 200]           # upper bounds


##############################################################################################
# the objective functions, for a single simulation and for a population

def prod_yield_calc_single(x_vector):
    xPR_local = xPR.copy()
    for idx, value in zip(param_indices, x_vector):
        xPR_local[idx] = value

    _, _, vP, pY = batch_cult_sim(base_params, hPR, xPR_local)
    return -vP, -pY


def prod_yield_calc_pop(pop):
    with mp.Pool(processes=mp.cpu_count()) as pool:
        res = pool.map(prod_yield_calc_single, pop)
    return np.array(res)


###########################################################################################
# the vectorised pymoo problem definitions

class SingleObjectiveProblem(Problem):
    def __init__(self, objective_index):
        super().__init__(
            n_var=len(param_indices),
            n_obj=1,
            xl=lower_bounds,
            xu=upper_bounds
        )
        self.obj_id = objective_index  # 0 → vP, 1 → pY
    def _evaluate(self, X, out, *args, **kwargs):
        F_all = prod_yield_calc_pop(X)
        out["F"] = F_all[:, self.obj_id].reshape(-1, 1)


class MultiObjectiveProblem(Problem):
    def __init__(self):
        super().__init__(
            n_var=len(param_indices),
            n_obj=2,
            xl=lower_bounds,
            xu=upper_bounds
        )
    def _evaluate(self, X, out, *args, **kwargs):
        out["F"] = prod_yield_calc_pop(X)


###########################################################################################
# running the optimisations

if __name__ == "__main__":

    # Single-objective vP
    print("Running single objective optimisation for vP")

    so_vP = SingleObjectiveProblem(objective_index=0)

    res_vP = minimize(
        so_vP,
        GA(pop_size=pop_size_single),
        termination=get_termination("n_gen", n_gens_single),
        seed=1,
        verbose=True
    )

    max_vP = -res_vP.F[0]
    print("Max vP:", max_vP)

    # Single-objective pY
    print("Running single objective optimisation for pY")

    so_pY = SingleObjectiveProblem(objective_index=1)

    res_pY = minimize(
        so_pY,
        GA(pop_size=pop_size_single),
        termination=get_termination("n_gen", n_gens_single),
        seed=1,
        verbose=True
    )

    max_pY = -res_pY.F[0]
    print("Max pY:", max_pY)

    problem = MultiObjectiveProblem()

    # full multiobjective
    res = minimize(
        problem,
        NSGA2(pop_size=pop_size_multi),
        termination=get_termination("n_gen", n_gens_multi),
        seed=1,
        verbose=True
    )

    pareto_X = res.X
    F = res.F

    vP_vals = -F[:, 0]
    pY_vals = -F[:, 1]

    print("Found Pareto points:", len(F))

    df = pd.DataFrame(pareto_X, columns=optimised_labels)
    df["vP"] = vP_vals
    df["pY"] = pY_vals

    df.to_csv(case_name+"_results.csv", index=False)
