"""
Main script for the Python Quantum Partial Differential Equation (QPDE) solver. The simulation parameters and equations are set up in a separate input file which is
imported when created and running the Simulation() class. The code uses SymPy to analytically derive expressions for the driver functions, derivatives of driver
functions, and auxiliary functions. This functionality is contained in the auto_sympy.py script which itself uses functions from custom_sympy.py. The integration 
subroutines were adapted from code by Ryan Vogt and Hunter Rouillard and are stored in kacewicz_functions. The Python packages NumPy, SciPy, and SymPy are required.
To run the code, install the necessary packages in your environment, change the import statement in the 'if (__name__=="__main__"):' function of this script (main.py),
and run the code using $ python main.py. The outputs will be saved in an output directory contained within the directory where main.py is, with the output directory
name listed in the input script.

Author: Conner Myers
Created: 11/14/23
"""

import os # Only used for creating the output directory if it does not already exist
import numpy as np
import sympy as sp
import math
import time
import kacewicz_functions
import auto_sympy

class Simulation():
    """ Initializes the main Simulation() object. The object stores simulation parameters, sets the initial and boundary conditions specified in the input deck, 
        initializes the simulation arrays for variables, derivatives, and auxiliary variables, and initializes the auto_sympy module for computing the driver
        functions and corresponding derivatives.
    Args:
        general_params (Python dict): Dictionary containing the the general parameters for the simulation. See a current input deck for the details on 
                                    the simulation parameters.
        sympy_params (Python dict): Dictionary containing the the SymPy parameters for the simulation, including variables, driver functions, and auxiliary functions.
                                    See a current input deck for the details on the simulation parameters.
    Returns:
        None
    """
    def __init__(self,general_params,sympy_params) -> None:
        self.N = general_params['N']
        self.xlims = general_params['xlims']
        self.xs = np.linspace(self.xlims[0], self.xlims[1], self.N, endpoint = True)
        self.h = (self.xlims[1] - self.xlims[0]) / (self.N - 1)
        self.r = general_params['taylor_r']
        self.num_ODEs = general_params['num_ODEs']
        self.num_aux = general_params['num_aux']

        # Custom time output
        self.t_outputs = general_params['time_outputs']
        self.t_output_length = len(self.t_outputs)
        self.t_counter = 0
        self.custom_outputs = np.zeros((self.num_ODEs,self.N,self.t_output_length))

        # Initialize value array
        self.U = np.zeros((self.num_ODEs,self.N))
        # Set values to specified initial conditions
        if (self.num_ODEs==1):
            if (general_params['initial_cond_type']=='fixed'):
                self.U[0] = general_params['initial_cond_vals']*np.ones(self.N)
            elif (general_params['initial_cond_type']=='specified'):
                self.U[0] = general_params['initial_cond_vals']
        else:
            for i in range(self.num_ODEs):
                if (general_params['initial_cond_type'][i]=='fixed'):
                    self.U[i] = general_params['initial_cond_vals'][i]*np.ones(self.N)
                elif (general_params['initial_cond_type'][i]=='specified'):
                    self.U[i] = general_params['initial_cond_vals'][i]
        # Set boundary conditions
        self.boundary_type = general_params['boundary_type']
        self.boundary_vals = general_params['boundary_vals']
        if (self.num_ODEs==1):
            if (self.boundary_type[0]=='fixed'):
                self.U[0,0] = self.boundary_vals[0]
            elif (self.boundary_type[0]=='open'):
                self.U[0,0] = 2*self.U[0,1] - self.U[0,2]
            if (self.boundary_type[1]=='fixed'):
                self.U[0,-1] = self.boundary_vals[1]
            elif (self.boundary_type[1]=='open'):
                self.U[0,-1] = 2*self.U[0,-2] - self.U[0,-3]
        else:
            for i in range(self.num_ODEs):
                if (self.boundary_type[i][0]=='fixed'):
                    self.U[i,0] = self.boundary_vals[i][0]
                elif (self.boundary_type[i][0]=='open'):
                    self.U[i,0] = 2*self.U[i,1] - self.U[i,2]
                if (self.boundary_type[i][1]=='fixed'):
                    self.U[i,-1] = self.boundary_vals[i][1]
                elif (self.boundary_type[i][1]=='open'):
                    self.U[i,-1] = 2*self.U[i,-2] - self.U[i,-3]

        self.aux = np.zeros((self.num_aux,self.N)) # auxiliary data arrays
        self.f_array = np.zeros((self.num_ODEs,self.r+1,self.N)) # array for storing f's and derivatives up to r-th order
        self.quantum = general_params['quantum']
        self.integral_mode = general_params['integral_mode']
        self.directory_name = general_params['directory_name']
        self.verbose = general_params['verbose']
        
        # Initialize time parameter hbar through CFL DΔt/Δx**2<0.5
        self.hbar = general_params['dt']

        self.epsilon = general_params['epsilon']
        self.N_tot = general_params['N_tot']
        (n,k) = kacewicz_functions.calculate_partition_parameters(self.epsilon,self.N_tot)
        self.n = n
        self.k = k
        # Create the integrand function, used in kacewicz_functions.integrate, based on the Taylor Polynomial order
        self.integrand_func = self._create_integrand_func()
        
        # Create TPoly Matrix for storing values from singular subinterval
        self.mat = np.zeros((self.num_ODEs,self.r+2,self.N-2))
        # Create TPoly Matrix
        self.ll = np.zeros((self.num_ODEs,self.r+2,self.N-2,self.n**(self.k-1)))
        # Create y matrix and set the initial y to the initial state and boundaries to boundaries (assuming fixed and equal at boundaries)
        self.y = np.zeros((self.num_ODEs,self.N,self.n+1))
        self.y[:,:,0] = (self.U)

        # Create auto_sympy object for equation evaluation
        self.evals = auto_sympy.equation_writer(sympy_params['list_equations'],sympy_params['list_auxiliary'],sympy_params['list_spatial_vars'],\
                                                sympy_params['list_constants'],sympy_params['temporal_var'],general_params['taylor_r'],general_params['index_range'])
        # Carry out a Finite Difference Approximation for evaluation of f,dfdt,d2fdt2,...
        # Evaluations of auxiliary function D also carried out via auto_sympy object
        self.evals.finite_diff_approx(z=sympy_params['list_spatial_vars'][0],dz_sym=sympy_params['list_constants'][0][0]) # no C aux

    """ Constructs the integrand function for the evaluation of integrals by the IntegrateGij() function. The order of the Taylor polynomial approximation is set
        in the input deck as 'r'. This function initializes the integrand_func() at the start and returns it as an attribute of the simulation() object.
    Args:
        None (self.r is set during Simulation.__init__())
    Returns:
        integrand_func (function): The integrand function which takes t (1d array), t0, and C (1 by r array of coefficients) to evaluate a driver function f 
            over a specified array of times in t.
    """
    def _create_integrand_func(self):
        # Creates the integrand function for integrating the ODEs, used as an input to kacewicz_functions.integrate
        def integrand_func(t,t0,C):
            func = 0
            for i in range(self.r+1):
                func += C[i]*(t-t0)**(i)/math.factorial(i)
            return func
        return integrand_func
    
    """ Creates the output directory if it doens't already exist and saves the initial output times, simulation time data, and grid data.
    Args:
        None
    Returns:
        None
    """
    def initial_save(self):
        # Creates directories, saves initial data
        # Create the output directory if it does not already exist
        if not os.path.exists(self.directory_name):
            os.makedirs(self.directory_name)
        if (self.quantum):
            self.sim_type = 'quantum'
        else:
            self.sim_type = 'classical'
        # Save the primary interval data
        with open(self.directory_name+'/'+self.sim_type+'_'+self.integral_mode+'_'+str(self.N)+'.npz','wb') as f:
            np.save(f,self.y)
        f.close()
        # Save the custom output data
        with open(self.directory_name+'/'+self.sim_type+'_'+self.integral_mode+'_'+str(self.N)+'_custom.npz','wb') as f:
            np.save(f,self.custom_outputs)
        f.close()
        # Save the primary time interval values (recalculated below using timestep hbar and primary interval size n**k-1)
        times = np.zeros(self.n+1)
        for i in range(self.n+1):
            times[i] = i*(self.n**(self.k-1))*self.hbar
        with open(self.directory_name+'/'+self.sim_type+'_'+self.integral_mode+'_'+str(self.N)+'_times'+'.npz','wb') as f:
            np.save(f,times)
        f.close()
        # Save spatial grid data
        with open(self.directory_name+'/'+self.sim_type+'_'+self.integral_mode+'_'+str(self.N)+'_dx'+'.npz','wb') as f:
            np.save(f,self.xs)
        f.close()

    """ Saves the simulation output data (simulation variables) for both standard (at each primary interval) and custom output times.
    Args:
        None
    Returns:
        None
    """
    def update_save(self):
        # Updates output files with new data
        # Save the primary interval data
        with open(self.directory_name+'/'+self.sim_type+'_'+self.integral_mode+'_'+str(self.N)+'.npz','wb') as f:
            np.save(f,self.y)
        f.close()
        # Save the custom output data
        with open(self.directory_name+'/'+self.sim_type+'_'+self.integral_mode+'_'+str(self.N)+'_custom.npz','wb') as f:
            np.save(f,self.custom_outputs)
        f.close()

    """ Sets the boundary values for the output data. This is not calculated automatically during the simulation since the regular gridpoint loops don't cover
        boundaries.
    Args:
        index (int): index for the ODE for which boundary points to evaluate
        primary interval (int): index for the primary interval for which to evaluate the boundary values of y
    Returns:
        None
    """
    def _set_boundary_values(self, index, primary_interval):
        if self.boundary_type[index][0] == 'fixed':
            self.y[index, 0, primary_interval + 1] = self.boundary_vals[index][0]
        elif self.boundary_type[index][0] == 'open':
            self.y[index, 0, primary_interval + 1] = 2 * self.y[index, 1, primary_interval + 1] - self.y[index, 2, primary_interval + 1]

        if self.boundary_type[index][1] == 'fixed':
            self.y[index, -1, primary_interval + 1] = self.boundary_vals[index][1]
        elif self.boundary_type[index][1] == 'open':
            self.y[index, -1, primary_interval + 1] = 2 * self.y[index, -2, primary_interval + 1] - self.y[index, -3, primary_interval + 1]

    """ Integrates the system of PDEs over all of the primary intervals. This function loops over each primary interval and saves the data at the end of each loop.
        If self.verbose is true, it will print out data on the runtime of each subroutine over the primary interval.
    Args:
        None
    Returns:
        None
    """
    def integratePDEs(self):
        # Create saving function, initial save data here
        self.initial_save()
        for i in range(self.n):
            t_start = i*self.n**(self.k-1)*self.hbar
            time_0 = time.time()
            self.BldTPoly()
            time_1 = time.time()
            self.IntegrateGij(t_start,i)
            time_2 = time.time()
            self.update_save() # Update output files
            time_3 = time.time()
            if (self.verbose):
                print("Completed primary interval " + str(i+1) + " out of " + str(self.n) + ".")
                print("Total primary interval time: " + str(time_3-time_0))
                print("BldTPoly time: " + str(time_1-time_0))
                print("IntegrateGij time: " + str(time_2-time_1))
                print("Update_save time: " + str(time_3-time_2))

    """ Loops over the secondary subintervals within a primary interval to construct the Taylor polynomial coefficient array (self.ll).
    Args:
        None
    Returns:
        None
    """
    def BldTPoly(self):
        # Loop over each subinterval and populate the taylor polynomial matrix (self.ll) for each secondary subinterval in the primary subinterval
        for i in range(self.n**(self.k-1)):
            # Calculate Derivatives
            self.Derivs()
            # Build Taylor Polynomial Matrix
            self.BldTMat()
            # Transfer over matrix values to ll
            for x in range(self.N-2):
                for k in range(self.num_ODEs):
                    for t in range(self.r+2):
                        self.ll[k,t,x,i] = self.mat[k,t,x]
            # Compute the values of self.U for the next subinterval
            self.NextInCond()
    
    """ Calculates values for the auxiliary functions, driver functions, and driver function derivatives up to order r at the current secondary subinterval using 
        the lambda functions for each from SymPy. All arrays are updated internally.
    Args:
        None
    Returns:
        None
    """
    def Derivs(self):
        # Create argument list for lambda functions
        function_arg_list = []
        # Indicies for function evaluation:
        gridpoint_indicies = np.arange(1,self.N-1)
        gridpoint_indicies_aux = np.arange(self.N)
        # Auxiliary functions:
        for aux2 in range(self.num_aux):
            function_arg_list.append(self.aux[aux2])
        # Functions and function derivatives:
        for eqn in range(self.num_ODEs):
            function_arg_list.append(self.U[eqn,:])
            for tylr in range(self.r):
                function_arg_list.append(self.f_array[eqn,tylr,:])

        # Boundary step
        if (self.num_ODEs==1):
            if (self.boundary_type[0]=='fixed'):
                self.U[0,0] = self.boundary_vals[0]
            elif (self.boundary_type[0]=='open'):
                self.U[0,0] = 2*self.U[0,1] - self.U[0,2]
            if (self.boundary_type[1]=='fixed'):
                self.U[0,-1] = self.boundary_vals[1]
            elif (self.boundary_type[1]=='open'):
                self.U[0,-1] = 2*self.U[0,-2] - self.U[0,-3]
        else:
            for i in range(self.num_ODEs):
                if (self.boundary_type[i][0]=='fixed'):
                    self.U[i,0] = self.boundary_vals[i][0]
                elif (self.boundary_type[i][0]=='open'):
                    self.U[i,0] = 2*self.U[i,1] - self.U[i,2]
                if (self.boundary_type[i][1]=='fixed'):
                    self.U[i,-1] = self.boundary_vals[i][1]
                elif (self.boundary_type[i][1]=='open'):
                    self.U[i,-1] = 2*self.U[i,-2] - self.U[i,-3]
        
        # Tolerance for rounding computations
        tolerance = 12 # using np.round, tolerance is an integer for the number of decimals to round to past the decimal

        # Auxiliary function evaluation
        for auxiliary in range(self.num_aux):
            # Note: auxiliary evaluation inputs are currently hard coded
            self.aux[auxiliary][gridpoint_indicies_aux] = self.evals.d_dict['d_'+str(auxiliary)+'_lambdified'](gridpoint_indicies_aux,*function_arg_list)
            self.aux[auxiliary][gridpoint_indicies_aux] = np.round(self.aux[auxiliary][gridpoint_indicies_aux],tolerance)
            if (self.boundary_type[auxiliary][0]=='open'):
                self.aux[auxiliary][0] = 2*self.aux[auxiliary][1] - self.aux[auxiliary][2]
            if (self.boundary_type[auxiliary][1]=='open'):
                self.aux[auxiliary][-1] = 2*self.aux[auxiliary][-2] - self.aux[auxiliary][-3]

        for eqn in range(self.num_ODEs):
            for tylr in range(self.r+1):
                self.f_array[eqn,tylr,gridpoint_indicies] = self.evals.f_dict['f'+str(eqn)+'_'+str(tylr)+'_lambdified'](gridpoint_indicies,*function_arg_list)
                self.f_array[eqn,tylr,gridpoint_indicies] = np.round(self.f_array[eqn,tylr,gridpoint_indicies],tolerance)
                if (self.boundary_type[eqn][0]=='open'):
                    self.f_array[eqn,tylr,0] = 2*self.f_array[eqn,tylr,1] - self.f_array[eqn,tylr,2]
                if (self.boundary_type[eqn][1]=='open'):
                    self.f_array[eqn,tylr,-1] = 2*self.f_array[eqn,tylr,-2] - self.f_array[eqn,tylr,-3]

    """ Builds the Taylor polynomial coefficient matrix U (driver function variables) and f_array (driver functions and their derivatives).
    Args:
        None
    Returns:
        None
    """
    def BldTMat(self):
        # Build Taylor Polynomial Matrix
        for k in range(self.num_ODEs):
            for x in range(self.N-2):
                self.mat[k,0,x] = self.U[k,x+1]
                for t in range(self.r+1):
                    self.mat[k,t+1,x] = self.f_array[k,t,x+1]

    """ Uses values from the Taylor polynomial coefficients to find the approximate value of U (field variables) for the start of the next secondary subinterval.
        The new values of U are needed to begin the process for constructing the Taylor polynomial coefficients for the next loop starting with Derivs().
    Args:
        None
    Returns:
        None
    """
    def NextInCond(self):
        # Determines the set of values for self.U for the next subinterval
        for k in range(self.num_ODEs):
            for x in range(self.N-2):
                TPoly_sum = 0.
                for t in range(self.r+2):
                    TPoly_sum += self.mat[k,t,x]*self.hbar**(t)/math.factorial(t)
                self.U[k,x+1] = TPoly_sum

    """ Integrates the PDEs over a primary subinterval to find the approximate solution (self.y) at the start of the next primary subinterval. This function is called
        after BldTPoly() for each primary loop. This is repeated until the final simulated time is reached. This function loops over each PDE, gridpoint, and secondary
        subinterval within a primary subinterval and integrates, saving approximate solutions y at each primary interval and custom output times
    Args:
        t_start (float): starting time of the current primary subinterval
        primary_interval (int): index of the current primary subinterval
    Returns:
        None
    """
    def IntegrateGij(self,t_start,primary_interval):
        # Integrate the driver function f (approximated via Taylor Polynomial terms) over all the secondary subintervals in a primary subinterval
        # Hard code n_samples for now
        n_samples = 2 # Only 2 is needed for 4-th order accurate in time when using Gauss quadrature
        M = self.n**(self.k-1)
        delta = 0.001
        for x in range(self.N-2):
            for k in range(self.num_ODEs):
                sum = 0
                for j in range(self.n**(self.k-1)):
                    t_old = t_start + (j*self.hbar)
                    t_new = t_start + ((j+1)*self.hbar)
                    for i in range(self.t_output_length):
                        if ((self.t_outputs[i]>=t_old)and(self.t_outputs[i]<t_new)):
                            self.custom_outputs[k,x+1,i] = kacewicz_functions.integrate(integrand_function=self.integrand_func,tlims=[t_old,t_new], \
                                                                                                     n_samples=n_samples,integral_mode=self.integral_mode, \
                                                                                                        C=self.ll[k,1:,x,j],t0=t_old,quantum=False,M=M, \
                                                                                                            delta=delta) + self.y[k,x+1,primary_interval] + sum

                    sum += kacewicz_functions.integrate(integrand_function=self.integrand_func,tlims=[t_old,t_new],n_samples=n_samples, \
                                                        integral_mode=self.integral_mode,C=self.ll[k,1:,x,j],t0=t_old,quantum=self.quantum,M=M,delta=delta)
                self.y[k,x+1,primary_interval+1] = self.y[k,x+1,primary_interval] + sum
                if (self.verbose):
                    print("Completed point " + str(x+1) + " out of " + str(self.N-2) + "...")
        # Set self.y boundary values to boundary values
        if (self.num_ODEs==1):
            if (self.boundary_type[0]=='fixed'):
                self.y[0,0,primary_interval+1] = self.boundary_vals[0]
            elif (self.boundary_type[0]=='open'):
                self.y[0,0,primary_interval+1] = 2*self.y[0,1,primary_interval+1] - self.y[0,2,primary_interval+1]
            if (self.boundary_type[1]=='fixed'):
                self.y[0,-1,primary_interval+1] = self.boundary_vals[1]
            elif (self.boundary_type[1]=='open'):
                self.y[0,-1,primary_interval+1] = 2*self.y[0,-2,primary_interval+1] - self.y[0,-3,primary_interval+1]
        else:
            for k in range(self.num_ODEs):
                if (self.boundary_type[k][0]=='fixed'):
                    self.y[k,0,primary_interval+1] = self.boundary_vals[k][0]
                elif (self.boundary_type[k][0]=='open'):
                    self.y[k,0,primary_interval+1] = 2*self.y[k,1,primary_interval+1] - self.y[k,2,primary_interval+1]
                if (self.boundary_type[k][1]=='fixed'):
                    self.y[k,-1,primary_interval+1] = self.boundary_vals[k][1]
                elif (self.boundary_type[k][1]=='open'):
                    self.y[k,-1,primary_interval+1] = 2*self.y[k,-2,primary_interval+1] - self.y[k,-3,primary_interval+1]


# Note: output values y currently not recorded during simulation
if (__name__=="__main__"):
    import inputs_1T as inputs

    quantum = inputs.general_param_dict['quantum']
    directory_name = inputs.general_param_dict['directory_name']
    integral_mode = inputs.general_param_dict['integral_mode']
    N = inputs.general_param_dict['N']

    my_sim = Simulation(inputs.general_param_dict,inputs.sympy_param_dict)
    print("SIMULATION PARAMETERS")
    print("\n")
    print("Total simulation time T: " + str(my_sim.hbar*my_sim.n**my_sim.k))
    print("dt calculated: " + str(my_sim.hbar))
    print("\n")
    print("number of primary intervals (n): " + str(my_sim.n))
    print("k: "+str(my_sim.k))
    print("Number of secondaries per primary (n**(k-1)): "+str(my_sim.n**(my_sim.k-1)))
    print("\n")
    
    start = time.time()
    my_sim.integratePDEs()
    end = time.time()
    print("Total runtime: " + str(end-start) + " seconds")

    directory_name = 'output_final__backup'
    
    if (quantum):
        sim_type = 'quantum'
    else:
        sim_type = 'classical'

    # Create the output directory if it does not already exist
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)

    # Save the temperature data
    with open(directory_name+'/'+sim_type+'_'+integral_mode+'_'+str(N)+'.npz','wb') as f:
        np.save(f,my_sim.y)
    f.close()

    # Save the custom output data
    with open(directory_name+'/'+sim_type+'_'+integral_mode+'_'+str(N)+'_custom.npz','wb') as f:
        np.save(f,my_sim.custom_outputs)
    f.close()

    # Save the primary time interval values (recalculated below using timestep hbar and primary interval size n**k-1)
    times = np.zeros(my_sim.n+1)
    for i in range(my_sim.n+1):
        times[i] = i*(my_sim.n**(my_sim.k-1))*my_sim.hbar
    with open(directory_name+'/'+sim_type+'_'+integral_mode+'_'+str(N)+'_times'+'.npz','wb') as f:
        np.save(f,times)
    f.close()

    # Save spatial grid data
    with open(directory_name+'/'+sim_type+'_'+integral_mode+'_'+str(N)+'_dx'+'.npz','wb') as f:
        np.save(f,my_sim.xs)
    f.close()
    
    # Example for loading the resulting data:
    #
    # import numpy as np
    # with open('{directory_name: default is 'output}/{file name: default from above is 'quantum_quad_101'}.npz', 'rb') as f:
    #     y = np.load(f)
    # print(y)