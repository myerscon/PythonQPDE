# Input file containing the equations and corresponding variables and constants to be solved

import sympy as sp
import numpy as np

##########################
### GENERAL PARAMETERS ###
##########################
directory_name = 'output_1T' # output directory name
verbose = True # Prints out status/diagnostics when true

# NUMERICAL PARAMETERS
taylor_r = 2
index_range = 3
integral_mode = 'quad' # integration scheme; "riemann" or "quad" supported

# PROBLEM PARAMETERS
xlims = (0.,1.)
N = 101
h_value = (xlims[1]-xlims[0])/(N-1)
CFL = 0.0008 # The timestep is constrained by instability at a larger CFL (we currently employ an explicit time integration scheme)
dt = CFL*(h_value**2)*3 # This expression comes the CFL condition ∆t < CFL*(∆x)**2/D
initial_cond_type = ('specified')
x_array = np.arange(0,1.01,0.01)
delta = 0.01
gaussian_array = 0.01+0.99*np.exp(-(x_array)/delta)
initial_cond_vals = np.array([(gaussian_array)**(4)]) # The material equation is in terms of T = temp**2 = sqrt(E)
boundary_type = ('fixed','open') # Dictates the boundary type at the left and right values
boundary_vals = (1.,None) # Left and right boundary values (fixed) -> If not fixed, use 'None' or an arbitrary value (unused)
num_ODEs = 1
num_aux = 1
# Quantum PDE solver pamaters:
quantum = False # Boolean value indicating whether to run the simulation in quantum or classical configurations
epsilon = 0.005 # Error tolerance
N_tot = 4500000 # Total subintervals

time_outputs = [0.0,0.005,0.01,0.02,0.04,0.08,0.12,0.16,0.20,0.24,0.28,0.32,0.36,0.54,0.72,0.9,1.08]

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

# Spatial and temporal variables
z = sp.Symbol("z")
t = sp.Symbol("t")

# Variable functions
#     -Each variable function should have a corresponding driver function
#     -The driver function and it's derivatives up to order r are used to approximate the solution
E = sp.Function("E")(z,t)

# Auxiliary functions - optional functions used to simplify expressions, coupling, etc
D = sp.Function("D")(z,t)

# Driver functions
f = (D*E.diff(z)).diff(z)

# Derivative of auxiliary functions - needed to substitute d(aux)/dt in Sympy expressions
D_expr = (E**(gamma/4))/3
D_deriv = (gamma/4)*(D/E)*sp.Derivative(E,t)

# List the constants serially in a single list.
dz_value = (xlims[1]-xlims[0])/(N-1)
list_constants = [[dz, gamma],[dz_value,3.5]]

# List the spatial variable and temporal variable. List the spatial variables together while keeping the temporal variable independent
list_spatial_vars = [z]
temporal_var = t

# List the equations (the number of variable/driver function pairs) for the auto_sympy script. Create a list of pairs of each variable and driver func
list_equations = [[E,f]]

# List the auxiliary functions. Create a list of pairs of each auxiliary function and its Sympy derivative expression (for substitution)
list_auxiliary = [[D,D_expr,D_deriv]]

# Sympy Parameter Dictionary
sympy_param_dict = {'list_equations':list_equations,'list_auxiliary':list_auxiliary,'list_spatial_vars':list_spatial_vars,\
                    'list_constants':list_constants,'temporal_var':temporal_var}