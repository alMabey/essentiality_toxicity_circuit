import sys
import os
directory = os.path.abspath(os.path.join('..'))
sys.path.append(directory)
import numpy as np
from matplotlib import pyplot as plt
from scripts.model_params import *
from scripts.cell_model_batch_culture import *
from scripts.batch_culture_simulation import *
import os
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination

# defining parameters
sS0 = 0.5; cultvol = 1.25 #quality of the nutrients and volume of the culture
vX = 726; kX = 1e3 # X export parameters
tmax = 20000; runintmax = 1e6 #simulation time parameters

N0  = 1e6
xS0 = 4.180555555555556e+22; # amount of glucose in media = 10g/L in 1.25L working vol in 3L vessel
M0 = 1e8 

#choosing the product transport method. 1 indicates this method is included
lin_trans      = 1
Tp_trans       = 0
T_trans        = 0
diff_trans     = 0

#mutaually exclusive; only one of these can be 1, rest must be 0
eprodtox       = 1
elongationtox  = 0

topology = [lin_trans, Tp_trans, T_trans, diff_trans, eprodtox, elongationtox]

base_params = [xS0, runintmax, tmax, N0, topology]

hPR0, xPR0 = model_params(sS0, vX, kX, cultvol, leaky_control=False)
hPR = np.array(hPR0)
xPR = np.array(xPR0)
# 0     1       2       3       4         5        6        7           8
#[w0,   wT,     wE,     wEp,    wTF,    wpTox,    wTp,     k_Ep,    Km_Ep,    
#   9      10             11                 12
# k_Tp,    Km_Tp, a_energy_pTox,     a_elongation_pTox, K_E, 
#   13      14      15       16       17   18     19
# # K_E   K_pTox, kdiffP, VolCell, VolCult, ksf, ksr]

wT = 20; wE = 20; wEp = 20; wTF = 20; wpTox = 2;
K_E = 0.3; K_pTox = 1.0

xPR[[1, 2, 3, 4, 5, 13, 14]] = [wT, wE, wEp, wTF, wpTox, K_E, K_pTox]

param_indices = [2, 3]             # e.g., wEp at index 2 (old behaviour)
# param_indices = [2, 3, 12]    # Example: optimize wEp, wTF, K_E
lower_bounds  = [0, 0]             # lower bounds for each parameter
upper_bounds  = [100, 100]           # upper bounds

n_params = len(param_indices)

def evaluate_model(x_vector):
    xPR_local = xPR.copy()
    for idx, value in zip(param_indices, x_vector):
        xPR_local[idx] = value

    vP, pY = batch_cult_prod_yield_calc(base_params, hPR, xPR_local)

    return -np.log(vP), -np.log(pY)

class SingleObjective(ElementwiseProblem):
    def __init__(self, obj_id):
        super().__init__(n_var=n_params,
                         n_obj=1,
                         xl=lower_bounds,
                         xu=upper_bounds)
        self.obj_id = obj_id

    def _evaluate(self, x, out):
        obj_vec = evaluate_model(x)
        out["F"] = obj_vec[self.obj_id]


class MultiObjectiveScaled(ElementwiseProblem):
    def __init__(self, scale):
        super().__init__(n_var=n_params,
                         n_obj=2,
                         xl=lower_bounds,
                         xu=upper_bounds)
        self.scale = scale

    def _evaluate(self, x, out):
        vP_log, pY_log = evaluate_model(x)
        out["F"] = np.array([vP_log, pY_log]) / self.scale

res_vP = minimize(
    SingleObjective(obj_id=0),
    GA(pop_size=30),
    seed=1,
    verbose=True
)
max_vP_log = res_vP.F[0]

res_pY = minimize(
    SingleObjective(obj_id=1),
    GA(pop_size=30),
    seed=1,
    verbose=True
)
max_pY_log = res_pY.F[0]

scale = np.array([abs(max_vP_log), abs(max_pY_log)])
termination = get_termination("n_gen", 80)
res = minimize(
    MultiObjectiveScaled(scale),
    NSGA2(pop_size=60),
    termination,
    seed=1,
    verbose=True
)

pareto_X = res.X                          # parameter sets
pareto_scaled = res.F                     # scaled objectives
pareto_unscaled = pareto_scaled * scale   # real objectives

vP_vals = np.exp(-pareto_unscaled[:, 0])
pY_vals = np.exp(-pareto_unscaled[:, 1])