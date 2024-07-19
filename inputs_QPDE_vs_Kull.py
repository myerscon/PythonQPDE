# Input file containing the equations and corresponding variables and constants to be solved

import sympy as sp
import numpy as np

##########################
### GENERAL PARAMETERS ###
##########################
directory_name = 'output_QPDE_vs_Kull' # output directory name
verbose = True # Prints out status/diagnostics when true

# NUMERICAL PARAMETERS
taylor_r = 2
index_range = 3
integral_mode = 'quad' # integration scheme; "riemann" or "quad" supported

# PROBLEM PARAMETERS
#xlims calculated below from x_array
N = 101
initial_cond_type = ('specified','specified')
x_array = 0.086258*np.arange(0,1.01,0.01)
xlims = (x_array[0],x_array[-1])
h_value = (xlims[1]-xlims[0])/(N-1)
CFL = 0.5
Diffusion = 69.8063
dt = CFL*(h_value**2)/Diffusion

# Source and initial (ambient) temperature
T_source = 0.15 # in keV
T_initial = 0.015 # in keV
 
# Length scale of Gaussian initial condition
Delta = 8.6258e-4 # in cm

# Initial Matter equation
T_ic_matter = T_initial + (T_source - T_initial)*np.exp(-x_array/Delta)
E_ic_radiation = (T_ic_matter)**4

initial_cond_vals = np.array([[E_ic_radiation],[T_ic_matter]])
boundary_type = (('fixed','open'),('fixed','open')) # Dictates the boundary type at the left and right values -> 'fixed', 'open'
boundary_vals = (((0.15)**4,None),(0.15,None)) # Left and right boundary values (fixed) -> If not fixed, use 'None' or an arbitrary value (unused)
num_ODEs = 2
num_aux = 1
# Quantum PDE solver pamaters:
quantum = False # Boolean value indicating whether to run the simulation in quantum or classical configurations
epsilon = 0.005 # Error tolerance
N_tot = 2250000 # Total subintervals

# Additional time output array: Saves data at selected times (simulation already saves output at the start, end, and primary interval points)
# Note: outputs must be spaced greater than the time interval dt used in the simulation
time_outputs = [0.0,0.0005,0.001,0.002,0.004,0.006,0.008,0.01,0.012]

# GENERAL PARAMETER DICTIONARY
general_param_dict = {'directory_name':directory_name,'verbose':verbose,'taylor_r':taylor_r,'index_range':index_range,'integral_mode':integral_mode,\
                      'xlims':xlims,'N':N,'dt':dt,'initial_cond_type':initial_cond_type,'initial_cond_vals':initial_cond_vals,'boundary_type':boundary_type,\
                      'boundary_vals':boundary_vals,'num_ODEs':num_ODEs,'num_aux':num_aux, 'quantum':quantum,'epsilon':epsilon,'N_tot':N_tot,\
                      'time_outputs':time_outputs}

########################
### SYMPY PARAMETERS ###
########################
# Constants
dz = sp.Symbol("dz")
gamma = sp.Symbol("gamma")
f_coupling = sp.Symbol("f_coupling")
g_coupling = sp.Symbol("g_coupling")
D_constant = sp.Symbol("D_constant")

# Spatial and temporal variables
z = sp.Symbol("z")
t = sp.Symbol("t")

# Variable functions
#     -Each variable function should have a corresponding driver function
#     -The driver function and it's derivatives up to order r are used to approximate the solution
E = sp.Function("E")(z,t)
T = sp.Function("T")(z,t)

# Auxiliary functions - optional functions used to simplify expressions, coupling, etc
D = sp.Function("D")(z,t)

# Driver functions
f = (D*E.diff(z)).diff(z) + f_coupling*((T**4-E)/T**gamma)

# Derivative of auxiliary functions - needed to substitute d(aux)/dt in Sympy expressions
g = -g_coupling*(T**4-E)/(T**gamma)

D_expr = (D_constant)*T**gamma
D_deriv = gamma*(D/T)*sp.Derivative(T, t)

# List the constants serially in a single list.
# dz should always be the first constant. It is calculated from xlims and N below.
dz_value = (xlims[1]-xlims[0])/(N-1)
list_constants = ([dz,gamma,f_coupling,g_coupling,D_constant],[dz_value,2.0,429.167,12.2909*1400,69.8063]) # Constants have units # Diffusion: 6.980652*10**3

# List the spatial variable and temporal variable. List the spatial variables together while keeping the temporal variable independent
list_spatial_vars = [z]
temporal_var = t

# List the equations (the number of variable/driver function pairs) for the auto_sympy script. Create a list of pairs of each variable and driver func
list_equations = [[E,f],[T,g]]

# List the auxiliary functions. Create a list of pairs of each auxiliary function and its Sympy derivative expression (for substitution)
list_auxiliary = [[D,D_expr,D_deriv]]

# Sympy Parameter Dictionary
sympy_param_dict = {'list_equations':list_equations,'list_auxiliary':list_auxiliary,'list_spatial_vars':list_spatial_vars,\
                    'list_constants':list_constants,'temporal_var':temporal_var}