import sys
import os
directory = os.path.abspath(os.path.join('..'))
sys.path.append(directory)
import numpy as np
from matplotlib import pyplot as plt
from scripts.model_params import *
from scripts.cell_model import *
from scipy.integrate import solve_ivp
import sksundae as sun


def batch_cult_sim(base_params, hPR, xPR, integration_method="BDF", plot=False, rtol=1E-6, atol=1E-9):

    #################################################################################################################################
    #importing base parameters
    xS0, runintmax, tmax, N0, topology = base_params
    #################################################################################################################################
    def rhsfn_ss(t, y, yp):
        yp[:] = BatchCultModel_SS(t, y, hPR, xPR, topology)

    #simulating to get initial conditions
    y0 = np.zeros(33)
    y0[0] = 0 # N0
    y0[1] = 1e4 # xS0
    y0[3] = 1e6 # iS0
    y0[4] = 1e6 # ee0
    y0[7] = 1e2 # pT0
    y0[10] = 1e2 # pE0
    y0[18] = 1e2 # pR0
    y0[22] = 1e6 # iP0
    y0[25] = 1e6 # pTF0

    #setting up the simulation and its hyperparameters
    solver = sun.cvode.CVODE(
        rhsfn_ss,
        method=integration_method,
        rtol=rtol,
        atol=atol,
        max_num_steps=200000     
    )

    #running the simulation
    t_eval = np.array([0, float(runintmax)])
    sol   = solver.solve(t_eval, np.asarray(y0, dtype=float))
    t = sol.t; y = sol.y                 
    # now creating initial conditions vector
    y_init = y[-1]
    #removing anything less than zero
    y_init[y_init<0] = 0

    #setting the initial number of bacteria, and substrate in the culture, and making suare external product+inducer are both 0
    y_init[0] = N0; # 1; % N0
    y_init[1] = xS0; # 1; % xS0
    y_init[2] = 0; # xP0

    #################################################################################################################################
    # defining the RHS wrapper for cvode
    def rhsfn(t, y, yp):
        yp[:] = BatchCultModel_DC(t, y, hPR, xPR, topology)

    #################################################################################################################################
    EPS = 1e-12
    # event to make the simulation end once xS hits or crosses zero
    def event_xS_depletes(t, y, events):
        events[0] = y[1] - EPS
    event_xS_depletes.terminal  = [True]
    event_xS_depletes.direction = [-1]

    #setting up the simulation and its hyperparameters
    solver = sun.cvode.CVODE(
        rhsfn,
        method=integration_method,
        rtol=rtol,
        atol=atol,
        eventsfn=event_xS_depletes,
        num_events=1,
        max_num_steps=200000     
    )

    #running the simulation
    t_eval = np.array([0, float(tmax)])
    sol   = solver.solve(t_eval, np.asarray(y_init, dtype=float))
    t = sol.t; y = sol.y                 
    y = np.array([y[:, j].copy() for j in range(y.shape[1])])
    return t, y

def batch_cult_prod_yield_calc(base_params, hPR, xPR):
    T, Y = batch_cult_sim(base_params, hPR, xPR)
    vProd = Y[2, -1]/T[-1]
    pYield = Y[2, -1]/Y[1, 0]

    if pYield > 1:
        ProdPerf = [np.nan, np.nan]
    else:
        ProdPerf = [vProd, pYield] 

    return ProdPerf[0], ProdPerf[1]
