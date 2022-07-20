# -*- coding: utf-8 -*-
"""
Date: 07.11.2020
@author: Pablo Merchán-Rivera
Topic: Morris method - Mass transport equation
"""

#%% Import libraries
import numpy as np
import time
from SALib.sample.morris import sample
from SALib.analyze import morris
from SALib.plotting.morris import horizontal_bar_plot, covariance_plot
import matplotlib.pyplot as plt


#%% Solver (mass transport equation)
def massTransport(M, a, xobs, R, n, D, v, lb, t):
    c1 = (M) / ((2*n*a*R) * np.sqrt((np.pi*D*t)/R))
    c2 = np.exp(-((xobs-(v*t))**2)/((4*D*t)/R))
    c3 = np.exp(-(lb*t))
    C  = c1 * c2 * c3
    return C


# %% Import observed values
observations = np.loadtxt('observed_values.txt')


# %% Setup problem

'''
R:  retardation coefficient [-]
n:  porosity [-]
D:  dispersion coefficient [m^2/d]
v:  average linear velocity [m/d]
lb: decay constant [1/d]
'''

problem = {'num_vars': 5,
           'names': ['R', 'n', 'D', 'v', 'lb'],
           'bounds': [[1.90, 2.10], [0.25, 0.35], [0.50, 0.70], [0.20, 0.25], [0.015, 0.020]]
           }

T = 20    # number of optimal trajectories to sample (between 2 and r) -- 10, 20, 30
p = 10    # p-level grid: number of grid levels (should be even) -- 4, 6, 8, 10
r = T+1   # r-trajectories: number of trajectories to generate
seed = 1  # random seed to reproduce the results

np.random.seed(seed)

# deterministic parameters
M = 200     # mass of solute [kg]
a = 1       # area [m^2]
xobs = 50   # observation point distance [m]

# time discretization
t0 = 1e-13                      # initial time ~ 0
tf = 735                        # final time
step = 1                        # number of time steps
t = np.arange(t0, tf, step)     # time steps to evaluate



# %% Run Morris method

param_values = sample(problem, N=r, num_levels=p, optimal_trajectories=T, seed=seed) 

Y = []
for ti in t:
    Yi = massTransport(M, a, xobs, param_values[:,0], param_values[:,1], param_values[:,2], param_values[:,3], param_values[:,4], ti)
    Y.append(Yi)
    
check_time = 180

Si = morris.analyze(problem, 
                    param_values,
                    Y[check_time], 
                    conf_level=0.95,
                    print_to_console=False,
                    num_levels=p, 
                    num_resamples=1000)


# %% Print and plot results

fig1, (ax1) = plt.subplots(1, 1)
horizontal_bar_plot(ax1, Si, {}, sortby='mu_star', unit=r"")
plt.title('Sensitivity ranking (t=%d)' % (check_time))

fig2, (ax2) = plt.subplots(1, 1)
covariance_plot(ax2, Si, {}, unit=r"")
plt.title('Covariance plot(t=%d)' % (check_time))
