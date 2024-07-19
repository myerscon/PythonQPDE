# Script for running PyPDE simulations of the 2T problem and saving, loading, and plotting the results

import os
import numpy as np
import matplotlib.pyplot as plt
import pprint
import time
from math import pi
from pde import PDE, CartesianGrid, MemoryStorage, ScalarField, plot_kymograph, plot_kymographs, PDEBase, FieldCollection, UnitGrid

# User supplied parameters
N = 50
directory_name = 'output_PyPDE_2T/T_amb=01/N='+str(N)
showfig = False
output_times = [0.0,0.005,0.01,0.02,0.04,0.08,0.12,0.16]

if not os.path.exists(directory_name):
    os.makedirs(directory_name)

class RadDiff2T_unitless(PDEBase):
    """Radiation Diffusion 2-Temperature (Unitless)"""
    def __init__(self,gamma=3.5,c_=0.01,bc="periodic"): # Must declare bcs
        self.gamma = gamma
        self.c_ = c_
        self.bc = bc

    def evolution_rate(self, state, t=0):
        """implement the python version of the evolution equation"""
        E, T = state
        
        grad_E = E.gradient(self.bc)[0]
        grad_T = T.gradient(self.bc)[0]
        
        E_t = (T**self.gamma/3)*(E.laplace(self.bc) + (self.gamma/T)*grad_T*grad_E) + ((T**4)-(E))/(T**self.gamma)
        T_t = -((T**4)-(E))/(self.c_*(T**self.gamma))

        return FieldCollection([E_t,T_t])

# Loop over output times
for final_t in output_times:
    # Initialization
    grid = CartesianGrid([[0, 1]], [N], periodic=False)
    initial_state_expression1 = "(0.9*exp(-x/0.01)+0.1)**4"
    initial_state_expression2 = "0.9*exp(-x/0.01)+0.1"
    expression_list = [initial_state_expression1,initial_state_expression2]
    state = FieldCollection.from_scalar_expressions(grid, expression_list)

    # Simulation parameters
    bc_left = {"value": 1.0}
    bc_right = {"derivative": 0.0}
    bc = [bc_left,bc_right]
    gamma = 3.5
    c_ = 0.1

    # solve the equation and store the trajectory
    storage = MemoryStorage()
    eq = RadDiff2T_unitless(gamma,c_,bc)
    start = time.time()
    result_2T = eq.solve(state, t_range=final_t, dt=0.0000002)
    end = time.time()
    print("Runtime: " + str(end-start))

    # Save generated output data
    dx_array = result_2T._grid._axes_coords[0]
    rad_E_array = result_2T._data_valid[0]
    mat_T_array = result_2T._data_valid[1]

    with open(directory_name+'/'+'dx_array_t='+str(final_t)+'.txt', 'wb') as f:
        np.savetxt(f,dx_array)
    f.close()
    with open(directory_name+'/'+'rad_E_array_t='+str(final_t)+'.txt', 'wb') as f:
        np.savetxt(f,rad_E_array)
    f.close()
    with open(directory_name+'/'+'mat_T_array_t='+str(final_t)+'.txt', 'wb') as f:
        np.savetxt(f,mat_T_array)
    f.close()

    # Load generated output data
    with open(directory_name+'/'+'dx_array_t='+str(final_t)+'.txt', 'rb') as f:
        dx_loaded = np.loadtxt(f)
    f.close()
    with open(directory_name+'/'+'rad_E_array_t='+str(final_t)+'.txt', 'rb') as f:
        rad_E_loaded = np.loadtxt(f)
    f.close()
    with open(directory_name+'/'+'mat_T_array_t='+str(final_t)+'.txt', 'rb') as f:
        mat_T_loaded = np.loadtxt(f)
    f.close()

    # Plot Result
    fig, axes = plt.subplots(1,1,figsize=(10,8))
    plt.title("PyPDE 2T t="+str(final_t))
    plt.ylabel("Temperature")
    plt.xlabel("Position")
    axes.plot(dx_loaded,rad_E_loaded**(1/4),label="PyPDE Radiation",color='C0')
    axes.plot(dx_loaded,mat_T_loaded,label="PyPDE Material",color='C1')
    axes.set_xlim([0.0,0.5])
    axes.set_ylim([0.0,1.05])
    plt.legend()
    if (showfig):
        plt.show()