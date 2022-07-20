# -*- coding: utf-8 -*-
"""
Date: 07.11.2020
@author: Pablo Merchán-Rivera
Topic: Morris method - Mass transport equation
"""

#%% Import libraries
import chaospy as cp
import matplotlib.pyplot as plt
import numpy as np
import time

# Timing
start = time.time()


#%% Solver (mass transport equation)
def massTransport(M, a, xobs, R, n, D, v, lb, t):
    c1 = (M) / ((2*n*a*R) * np.sqrt((np.pi*D*t)/R))
    c2 = np.exp(-((xobs-(v*t))**2)/((4*D*t)/R))
    c3 = np.exp(-(lb*t))
    C  = c1 * c2 * c3
    return C


# %% Import observed values
observations = np.loadtxt('observed_values.txt')


#%% Model parameters

# sampling settings
numSamples = 1     # number of samples
numParameters = 5   # number of parameters
counter = 0         # initiate the counter to track the number of model evaluations

# deterministic parameters
M = 200     # mass of solute [kg]
a = 1       # area [m^2]
xobs = 50   # observation point distance [m]

# stochastic parameters
np.random.seed(1234)
R_distro  = cp.Uniform(1.90, 2.10)      # retardation coefficient [-]
n_distro  = cp.Uniform(0.25, 0.35)      # porosity [-]
D_distro  = cp.Uniform(0.50, 0.70)      # dispersion coefficient [m^2/d]
v_distro  = cp.Uniform(0.20, 0.25)      # average linear velocity [m/d]
lb_distro = cp.Uniform(0.015, 0.020)    # decay constant [1/d]

# join distribution and samples
joint_distro = cp.J(R_distro, n_distro, D_distro, v_distro, lb_distro)
joint_samples = joint_distro.sample(size = numSamples)

# time discretization
t0 = 1e-13                      # initial time ~ 0
tf = 735                        # final time
step = 4                        # number of time steps
t = np.arange(t0, tf, step)     # time steps to evaluate


# %% Run model evaluations

# storage array to save results
foo = np.zeros((len(t)))

# set deterministic parameters
R =   1.90
n =   0.25
D =   0.50
v =   0.20
lb =  0.015

for j in range(len(t)):
    # calculate solute concentration [kg/m^3]
    foo[j] = massTransport(M, a, xobs, R, n, D, v, lb, t[j])

day_interest = 200
t_interest = [int(day_interest/step)]

       
# %% Plot results

plt.figure('Solute Concentration (MC samples: %d)' % (numSamples))

plt.plot(t, foo, 'darkorange', alpha=0.99, label='deterministic realization')
plt.scatter(observations[:,0], observations[:,1], c='purple', marker='o', s=5, zorder=10)
plt.ylabel('Concentration [kg/m$^3$]')
plt.xlabel("Time [days]")
plt.xlim([0, tf])
plt.title('Solute Concentration (number of samples: %d)' % (numSamples))
plt.legend()

# stop timer
stop = time.time() 
elapsed_time = stop - start

# print results
print("Total elapsed time: " + str(round(elapsed_time/60, 2)) + " minutes")