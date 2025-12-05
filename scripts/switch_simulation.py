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


def batch_cult_sim_switch(base_params, hPR, xPR_pre, xPR_post, integration_method="BDF", plot=False, rtol=1E-6, atol=1E-9):

    #################################################################################################################################
    #importing base parameters
    xS0, runintmax, tmax, tswitch, N0, topology = base_params
    #################################################################################################################################
    #simulating to get initial conditions
    y0 = np.zeros(36)
    y0[0] = 0 # N0
    y0[1] = 1e4 # xS0
    y0[3] = 1e6 # iS0
    y0[4] = 1e6 # ee0
    y0[7] = 1e2 # pT0
    y0[10] = 1e2 # pE0
    y0[21] = 1e2 # pR0
    y0[25] = 1e6 # iP0
    y0[28] = 1e6 # pTF0

    t_eval = np.linspace(0.0, float(runintmax), 1000)

    sol = solve_ivp(
        lambda t, y: BatchCultModel_SS(t, y, hPR, xPR_pre, topology),
        (0.0, float(runintmax)),
        y0,
        t_eval=t_eval,
        method="BDF",
        rtol=1e-6,
        atol=1e-6
    )

    t_sol = sol.t
    y_sol = sol.y           

    # now creating initial conditions vector
    y_init = y_sol[:,-1]
    #removing anything less than zero
    y_init[y_init<0] = 0

    #setting the initial number of bacteria, and substrate in the culture, and making suare external product+inducer are both 0
    y_init[0] = N0; # 1; % N0
    y_init[1] = xS0; # 1; % xS0
    y_init[2] = 0; # xP0

    #################################################################################################################################
    # defining the RHS wrapper for cvode
    def rhsfn_pre(t, y, yp):
        yp[:] = BatchCultModel_DC(t, y, hPR, xPR_pre, topology)

    def rhsfn_post(t, y, yp):
        yp[:] = BatchCultModel_DC(t, y, hPR, xPR_post, topology)

    #################################################################################################################################
    #now simulating from induction

    EPS = 1e-12
    # event to make the simulation end once xS hits or crosses zero
    def event_xS_depletes(t, y, events):
        events[0] = y[1] - EPS
    event_xS_depletes.terminal  = [True]
    event_xS_depletes.direction = [-1]

    #setting up the simulation and its hyperparameters
    solver_pre = sun.cvode.CVODE(
        rhsfn_pre,
        method=integration_method,
        rtol=rtol,
        atol=atol,
        eventsfn=event_xS_depletes,
        num_events=1,
        max_num_steps=200000     
    )

    #running the simulation
    t_eval_pre = np.array([0, float(tswitch)])
    sol_pre   = solver_pre.solve(t_eval_pre, np.asarray(y_init, dtype=float))
    t_pre = sol_pre.t; y_pre = sol_pre.y                 
    y_pre = np.array([y_pre[:, j].copy() for j in range(y_pre.shape[1])])
    y_switch = y_pre[:, -1]; switch_idx = len(t_pre)

    solver_post = sun.cvode.CVODE(
        rhsfn_post,
        method=integration_method,
        rtol=rtol,
        atol=atol,
        eventsfn=event_xS_depletes,
        num_events=1,
        max_num_steps=200000     
    )
    t_eval_post = np.array([float(tswitch), tmax])
    sol_post   = solver_post.solve(t_eval_post, np.asarray(y_switch, dtype=float))
    t_post = sol_post.t; y_post = sol_post.y                 
    y_post = np.array([y_post[:, j].copy() for j in range(y_post.shape[1])])

    T = np.concatenate([t_pre, t_post]); Y = np.concatenate([y_pre, y_post], axis=1)

    return T, Y, switch_idx