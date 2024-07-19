# Takes inputs of constants, auxiliary functions, and functions and creates an object with discretized approximations
import sympy as sp
import custom_sympy

# takes inputs from input file (or other method) and creates discretized functions to be used within the simulation function
# to be used in place of evaluations.py
class equation_writer():
    """ Initializes the equation_writer object by setting object attributes both specified in the input deck and calculated using internal functions.
    Args:
        self
        equation_list (list of list of Sympy variables): A list of lists of 2 Sympy variables. There is one '2-list' for each equation in the input file,
            and each '2-list' contains first the variable for that equation and second the driver function for that equation.
        auxiliary_list (list of list of Sympy variables): A list of lists of 3 Sympy variables. There is one '3-list' for each auxiliary in the input file,
            and each '3-list' contains first the variable for that auxiliary, the Sympy expression for that auxiliary, and the derivative of the Sympy expression.
        spatial_vars_list (list of list of Sympy variables): A list of lists of Sympy spatial variables
        constants_list (list of list of Sympy variables): A list of lists of Sympy constants
        temporal_var (Sympy variable): Sympy variable for the temporal variable in the equations (t)
        taylor_r (int): integer corresponding to the highest derivative of the taylor series used in the solver (provided in input deck)
        index_range (int): specifies the range of the discretized (in space) functions, ranging from -(index_range) to +(index_range).
                            -> For example, an index range of 3 leads to a arrays of length 7: (j-3,j-2,j-1,j,j+1,j+2,j+3)
        """
    def __init__(self,equation_list,auxiliary_list,spatial_vars_list,constants_list,temporal_var,taylor_r=2,index_range=3) -> None:
        self.taylor_r = taylor_r
        self.index_range = index_range
        self.t = temporal_var
        self.equation_list = equation_list
        [self.string_equations,self.num_equations] = self._create_equation_list(equation_list)
        self.auxiliary_list = auxiliary_list
        [self.string_auxiliary,self.num_auxiliary] = self._create_auxiliary_list(auxiliary_list)
        self.spatial_vars_list = spatial_vars_list
        [self.string_spatial_vars,self.num_spatial_vars] = self._create_spatial_variable_list(spatial_vars_list)
        self.constants_list = constants_list
        [self.string_constants,self.num_constants] = self._create_constants_list(constants_list)
        self.equation_index = self._create_equation_index_list(temporal_var,self.string_equations,self.num_equations,taylor_r,index_range)
        self.auxiliary_index = self._create_auxiliary_index_list(temporal_var,self.string_auxiliary,self.num_auxiliary,auxiliary_list,self.string_equations,self.num_equations,equation_list,self.equation_index,index_range)
        self.disc_var_list = self._create_discretized_var_list(self.string_equations,self.num_equations,self.string_auxiliary,self.num_auxiliary,taylor_r)


    """ Takes the input parameter 'equation_list' and returns a list of name of each equation and the number of equations.
    Args:
        self
        equation_list (list of list of Sympy variables): A list of lists of 2 Sympy variables. There is one '2-list' for each equation in the input file,
            and each '2-list' contains first the variable for that equation and second the driver function for that equation.
    Returns:
        string_equations (list of strings): A list of strings for each equation variable in the input deck (does not inlcude driver functions)
        num_equations (int): An integer corresponding to the number of equations
    """
    def _create_equation_list(self,equation_list):
        string_equations = []
        num_equations = len(equation_list)
        for eqn in range(num_equations):
            string_list = []
            string_list.append(equation_list[eqn][0].name) # no need for name creation for f, only E
            string_equations.append(string_list)

        return [string_equations,num_equations]


    """ Takes the input parameter 'auxiliary_list' and returns a list of name of each auxiliary function and the number of auxiliary functions.
    Args:
        self
        auxiliary_list (list of list of Sympy variables): A list of lists of 3 Sympy variables. There is one '3-list' for each auxiliary in the input file,
            and each '3-list' contains first the variable for that auxiliary, the Sympy expression for that auxiliary, and the derivative of the Sympy expression.
    Returns:
        string_auxiliary (list of strings): A list of strings for each auxiliary variable in the input deck (does not inlcude expressions/derivatives)
        num_auxiliary (int): An integer corresponding to the number of auxiliaries
    """ 
    def _create_auxiliary_list(self,auxiliary_list):
        string_auxiliary = []
        num_auxiliary = len(auxiliary_list)
        for aux in range(num_auxiliary):
            string_list = []
            string_list.append(auxiliary_list[aux][0].name) # do not need names for auxiliary derivatives
            string_auxiliary.append(string_list)

        return [string_auxiliary,num_auxiliary]
    

    """ Takes the input parameter 'spatial_vars_list' and returns a list of name of each spatial variable and the number of spatial variables.
    Args:
        self
        spatial_vars_list (list of list of Sympy variables): A list of lists of Sympy spatial variables
    Returns:
        string_spatial_vars (list of strings): A list of strings for Sympy spatial variables in the input deck
        num_spatial_vars (int): An integer corresponding to the number of Sympy spatial variables
    """ 
    def _create_spatial_variable_list(self,spatial_vars_list):
        string_spatial_vars = []
        num_spatial_vars = len(spatial_vars_list)
        for var in range(num_spatial_vars):
            string_spatial_vars.append(spatial_vars_list[var].name)

        return [string_spatial_vars,num_spatial_vars]
    

    """ Takes the input parameter 'constants_list' and returns a list of name of each constant and the number of constant.
    Args:
        self
        constants_list (list of list of Sympy variables): A list of lists of Sympy constants
    Returns:
        string_constants (list of strings): A list of strings for Sympy constant in the input deck
        num_constants (int): An integer corresponding to the number of Sympy constant
    """ 
    def _create_constants_list(self,constants_list):
        string_constants = []
        num_constants = len(constants_list[0])
        for const in range(num_constants):
            string_constants.append(constants_list[0][const].name)

        return [string_constants,num_constants]
    

    """ Creates a Python dictionary containing an array of (discretized in space, continuous in time) equation variables for each equation variable
            in the input deck. The resulting 'equation_index' dictionary enables the Finite Difference Approximation of each variable (initially
            continuous in both space and time) by providing an array of each variable at discretized points in space. This function also provides
            an array of derivatives of each variable discretized in space, used in calculating the Finite Differenced form of derivatives of
            equations in the input deck.
    Args: 
        self
        t (Sympy variable): Sympy variable t (time variable) declared in the input deck.
        string_equations (list of strings): list of strings for the name of each variable (computed in _create_equation_list())
        num_equations (int): integer corresponding to the number of equations (computed in _create_equation_list())
        taylor_r (int): integer corresponding to the highest derivative of the taylor series used in the solver (provided in input deck)
        index_range (int): specifies the range of the discretized (in space) functions, ranging from -(index_range) to +(index_range).
                            -> For example, an index range of 3 leads to a arrays of length 7: (j-3,j-2,j-1,j,j+1,j+2,j+3)
    Returns:
        equation_index (Python dictionary): The returned dictionary contains arrays of Sympy functions that are continuous is time but not in space.
            Within the dictionary, there is an array of the Sympy functions for each equation variable, and the derivatives of each equation variable
            up to derivatives of order taylor_r specified in the input. The keys in the dictionary for each array correspond to the name of the variable
            followed by the number '0' for the base variable array or '1' for the first derivative of the variable, '2' for the second derivative, and
            so on. Ultimately, the 'equation_index' dictionary that is created by this function is used in calculating the Finite Difference Approximation
            in the function finite_diff_approx().
    """
    def _create_equation_index_list(self,t,string_equations,num_equations,taylor_r,index_range):
        # Creates an indexed list of functions from an input function.
        #     -Needed for approximating spatial derivatives with custom finite difference method approximation functions (custom_sympy.py)
        equation_index = {}
        for eqn in range(num_equations):
            for tylr in range(taylor_r+1):
                next_list = []
                next_list_name = string_equations[eqn][0] + str(tylr)
                for idx in range(2*(index_range)+1):
                    if (idx<(index_range)):
                        next_list.append(sp.Function(next_list_name+'_{j-'+str(index_range-idx)+'}')(self.t))
                    elif (idx>(index_range)):
                        next_list.append(sp.Function(next_list_name+'_{j+'+str(idx-index_range)+'}')(self.t))
                    else:
                        next_list.append(sp.Function(next_list_name+'_{j}')(self.t))
                equation_index[next_list_name] = next_list

        return equation_index
    

    """ This function is nearly identical to the analog above for equations, _create_equation_index_list(). It generally carries out the same procedures 
            but for the auxiliary functions instead of for the regular equations but with additional steps. Read the description under _create_equation_index_list()
            for more detail on the first half of the function. The function creates an auxiliary index containing arrays of the auxiliary functions that are
            continuous in time but discretized in space for use in the Finite Difference Approximations. However, it also creates arrays of the derivatives of
            auxiliary functions (the derivatives of the auxiliary functions are specified by the user in the input deck). The derivatives of the auxiliary functions
            are needed to substitute into the equations they appear in when calculating derivatives. 
    Args:
        self
        t (Sympy variable): Sympy variable t (time variable) declared in the input deck.
        string_auxiliary (list of strings): list of strings for the name of each auxiliary function (computed in _create_auxiliary_list())
        num_auxiliary (int): integer corresponding to the number of auxiliary function (computed in _create_auxiliary_list())
        auxiliary_list (list of Sympy variables): A list of lists of 3 Sympy variables. There is one '3-list' for each auxiliary in the input file,
            and each '3-list' contains first the variable for that auxiliary, the Sympy expression for that auxiliary, and the derivative of the Sympy expression.
        string_equations (list of strings): list of strings for the name of each variable (computed in _create_equation_list())
        num_equations (int): integer corresponding to the number of equations (computed in _create_equation_list())
        equation_list (list of list of Sympy variables): A list of lists of 2 Sympy variables. There is one '2-list' for each equation in the input file,
            and each '2-list' contains first the variable for that equation and second the driver function for that equation.
        equation_index (Python dictionary): Output of _create_equation_index_list(). See the above function and associated documentation for more details.
        index_range (int): specifies the range of the discretized (in space) functions, ranging from -(index_range) to +(index_range).
                            -> For example, an index range of 3 leads to a arrays of length 7: (j-3,j-2,j-1,j,j+1,j+2,j+3)
    Returns:
        auxiliary_index (Python dictionary): The returned dictionary contains arrays of Sympy functions that are continuous is time but not in space.
            Within the dictionary, there is an array of the Sympy functions for each auxiliary function and the derivatives of each auxiliary function.
            The keys in the dictionary for each array correspond to the name of the auxiliary function, and the name of each auxiliary function followed by
            '_deriv' in the case of derivatives of auxiliary functions. The 'auxiliary_index' dictionary is used in calculating the derivatives of each
            equation in the function finite_diff_approx().
    """
    def _create_auxiliary_index_list(self,t,string_auxiliary,num_auxiliary,auxiliary_list,string_equations,num_equations,equation_list,equation_index,index_range):
        # Creates an indexed list of auxiliary functions from an input function.
        #     -Needed for approximating spatial derivatives with custom finite difference method approximation functions (custom_sympy.py)
        auxiliary_index = {}
        for aux in range(num_auxiliary):
            # create auxiliary index dictionary
            next_list = []
            next_list_name = string_auxiliary[aux][0]
            for idx in range(2*(index_range)+1):
                if (idx<(index_range)):
                    next_list.append(sp.Function(next_list_name+'_{j-'+str(index_range-idx)+'}')(self.t)) # j-1,j-2,...
                elif (idx>(index_range)):
                    next_list.append(sp.Function(next_list_name+'_{j+'+str(idx-index_range)+'}')(self.t)) # j+1,j+2,...
                else:
                    next_list.append(sp.Function(next_list_name+'_{j}')(self.t)) # j
            auxiliary_index[next_list_name] = next_list
        # create auxiliary derivative index dictionary
        #     now that we have dictionaries containing all variables and auxiliary variables, we can create a list of expressions containing
        #     the derivatives of auxiliary variables at each index point
        for aux in range(num_auxiliary):
            deriv_list = []
            deriv_list_name = string_auxiliary[aux][0] + '_deriv'
            deriv = auxiliary_list[aux][2]
            for idx in range(2*(index_range)+1):
                deriv_k = deriv
                for eqn in range(num_equations):
                    deriv_k = deriv_k.subs(equation_list[eqn][0],equation_index[string_equations[eqn][0]+'0'][idx])
                for aux in range(num_auxiliary):
                    deriv_k = deriv_k.subs(auxiliary_list[aux][0],auxiliary_index[string_auxiliary[aux][0]][idx])
                deriv_list.append(deriv_k)
            auxiliary_index[deriv_list_name] = deriv_list

        return auxiliary_index
    
    
    """ Creates a list of discretized Sympy variables for each variable (including from each equation and auxiliary function). The discretized Sympy variables
        allow for the creation of lambda functions in Sympy which can be called in Python to calculate the functions. The discretized variables are an essential
        step in converting the continuous in time, discontinuous in space Sympy functions into functions that are discretized in both space and time and can be
        used to advanced the simulation forward in time to find the solution. The discretized Sympy variables are named by adding a '_' to the end of each variable
        name. An index variable 'j' is also created to be used in the discretized form of the equations.
        Args:
            self
            string_equations (list of strings): list of strings for the name of each variable (computed in _create_equation_list())
            num_equations (int): integer corresponding to the number of equations (computed in _create_equation_list())
            string_auxiliary (list of strings): list of strings for the name of each auxiliary function (computed in _create_auxiliary_list())
            num_auxiliary (int): integer corresponding to the number of auxiliary function (computed in _create_auxiliary_list())
            taylor_r (int): integer corresponding to the highest derivative of the taylor series used in the solver (provided in input deck)
        Returns:
            disc_var_list (list of Sympy variables): A list of Sympy variables, starting with the Sympy index variable 'j', followed by each equation and auxiliary
                variable with an added '_' (indicating to us that it is a discretized form of each equation)
        """
    def _create_discretized_var_list(self,string_equations,num_equations,string_auxiliary,num_auxiliary,taylor_r):
        # Creates discretized variables for each function and auxiliary function for final code printing
        disc_var_list = {}
        disc_var_list['j'] = sp.Symbol("j") # index variable
        for eqn in range(num_equations):
            for tylr in range(taylor_r+1):
                disc_var_list[string_equations[eqn][0]+str(tylr)+'_'] = sp.IndexedBase(string_equations[eqn][0]+str(tylr))
        for aux in range(num_auxiliary):
            disc_var_list[string_auxiliary[aux][0]+'_'] = sp.IndexedBase(string_auxiliary[aux][0])

        return disc_var_list
    
    
    """ Approximates equations and their derivatives using the Finite Difference Method and develops Sympy lambda functions that can be called from the 
        equation_writer() object. For each equation, equation derivative up to order taylor_r, and auxiliary function, three outputs are stored: The Sympy
        expression (symbolic), the discretized Sympy expression (a numerical relation), and a Sympy lambda function (callable object). Each output is from
        the same equation.
        
        This set of three outputs are stored in Python dictionaries labeled 'f_dict' for equations and their derivatives, or 'd_dict' for the auxiliary functions. 
        Each output is stored in the relevant Python dictionary with a key (string), with the letter 'f' for equations/derivatives and 'd' for auxiliaries. An example
        of the resulting dictionary for an input deck with 1 equations, 1 auxiliary functions, and taylor_r=2 is shown below:

        f_dict = {
            'f0_0':             Symbolic expression of equation f
            'f0_0_indexed':     Indexed Sympy expression for equation f
            'f0_0_lambdified':  Sympy lambda function for equation f (callable Python function)
            'f0_1':             Symbolic expression of first derivative of equation f, dfdt
            'f0_1_indexed':     Indexed Sympy expression for first derivative of equation f, dfdt
            'f0_1_lambdified':  Sympy lambda function for for first derivative of equation f, dfdt (callable Python function)
            'f0_2':             Symbolic expression of secon derivative of equation f, d2fdt2
            'f0_2_indexed':     Indexed Sympy expression for second derivative of equation f, d2fdt2
            'f0_2_lambdified':  Sympy lambda function for for second derivative of equation f, d2fdt2 (callable Python function)
        }

        d_dict = {
            'd0':               Symbolic expression of auxiliary function d
            'd0_indexed':       Indexed Sympy expression for auxiliary function d
            'd0_lambdified':    Sympy lambda function for auxiliary function d (callable Python function)
        }

        finite_diff_approx() creates the Python dictionaries then stores them on the equation_writer object. In the main Python script, the equation_writer()
        object is initiated first, and then the finite_diff_approx() method is called on that object:

            from auto_sympy import equation_writer
            eq = equation_writer({inputs here})
            eq.finite_diff_approx({inputs here})

        The lmabda functions can then be called using:

            calculated_array = eq.f_dict['f0_0_lambdified']({lambda function inputs})

        Currently, the inputs to every Sympy lambdified function is the gridpoint indicies for calculating the function for the first argument, followed by the
        values for all of the auxiliary functions (in order), followed by all of the equations and their derivatives, in order. For example, if we have one
        equation (f), one auxiliary function (d), and taylor_r=2, the inputs should look like:

            (gridpoint_indicies,d_values,f_values,dfdt_values,d2fdt2_values)
        
        Examine how the lambda functions are used in main.py for more information (under the function Derivs). Additionally, to understand the equation_writer()
        object and the Python dictionaries resulting from a set of inputs and the finite_diff_approx() function, it may be helpful to print out all of the 
        equation_writer() object attributes using:
        
            from pprint import pprint
            eq = equation_writer.({inputs})
            print(vars(eq))

        Args:
            self
            z (Sympy symbol): Sympy spatial variable for which the Finite Difference Approximation is carried out
            dz_sym (Sympy symbol): Sympy constant corresponding to the grid size in the Finite Difference Approximation
            direction (string): Corresponds to the type of Finite Difference Approximation, including 'central', 'forward', 'back'
        Returns:
            None
        """
    def finite_diff_approx(self,z,dz_sym,direction='central'):
        # Replaces spatial derivatives with finite difference approximations
        # Note: You may need to modify/extend the finite differencing steps in this function depending on your driver function.
        #       Make sure that each order of derivative of functions and auxiliary functions are present in the finite differencing step.
        # Compute f
        index_range = self.index_range
        t = self.t

        # Create lambdify arg list - all lambda functions take all other functions as inputs
        lambdify_arg_list = [self.disc_var_list['j']]
        for aux in range(self.num_auxiliary):
            lambdify_arg_list.append(self.disc_var_list[self.string_auxiliary[aux][0]+'_'])
        for eqn in range(self.num_equations):
            for tylr in range(self.taylor_r+1):
                lambdify_arg_list.append(self.disc_var_list[self.string_equations[eqn][0]+str(tylr)+'_'])

        # Lambdify auxiliary functions
        d_dict = {}

        var_index = self.disc_var_list['j']
        # Assuming no derivatives in auxiliary functions
        for aux in range(self.num_auxiliary):
            # f_D is the auxiliary function expression for each aux function
            f_D = self.auxiliary_list[aux][1] # D expression
            # Save f_D expression for debugging
            d_dict['d_'+str(aux)] = f_D
            # Now loop over each equation variable for substitution in f_D in case they show up in f_D
            for eqn in range(self.num_equations):
                E = self.equation_list[eqn][0] # primary variable for each equation
                E_j_list = self.equation_index[self.string_equations[eqn][0]+'0']
                f_D = f_D.subs(E,E_j_list[index_range]) # replace any instances of variable E in driver function with E_j -> assumes no derivatives in f_D
                for tylr in range(self.taylor_r+1):
                    E_index = self.disc_var_list[self.string_equations[eqn][0]+str(tylr)+'_']
                    E_list = self.equation_index[self.string_equations[eqn][0]+str(tylr)]
                    f_D = custom_sympy.index_subs(f_D,E_list,E_index,var_index,self.index_range)
            for constant in range(self.num_constants):
                f_D = f_D.subs(self.constants_list[0][constant],self.constants_list[1][constant])
            d_dict['d_'+str(aux)+'_indexed'] = f_D
            d_dict['d_'+str(aux)+'_lambdified'] = sp.lambdify(lambdify_arg_list,f_D)

        setattr(self,'d_dict',d_dict)

        f_dict = {}

        for eqn in range(self.num_equations):
            f = self.equation_list[eqn][1] # driver function for that variable
            for eqn2 in range(self.num_equations):
                E = self.equation_list[eqn2][0] # primary variable for each equation
                E_j_list = self.equation_index[self.string_equations[eqn2][0]+'0']
                f = f.subs(sp.Derivative(E,(z,2)),custom_sympy.second_diff_sym(E,E_j_list,dz_sym,direction='central')) # approximate second derivatives
                f = f.subs(sp.Derivative(E,z),custom_sympy.first_diff_sym(E,E_j_list,dz_sym,direction='central')) # approximate first derivatives
                f = f.subs(E,E_j_list[index_range]) # replace any instances of variable E in driver function with E_j
            for aux in range(self.num_auxiliary):
                D = self.auxiliary_list[aux][0] # primary variable for each equation
                D_j_list = self.auxiliary_index[self.string_auxiliary[aux][0]]
                f = f.subs(sp.Derivative(D,(z,2)),custom_sympy.second_diff_sym(D,D_j_list,dz_sym,direction='central')) # approximate second derivatives
                f = f.subs(sp.Derivative(D,z),custom_sympy.first_diff_sym(D,D_j_list,dz_sym,direction='central')) # approximate first derivatives
                f = f.subs(D,D_j_list[index_range]) # replace any instances of variable D in driver function with D_j
            
            f_dict['f'+str(eqn)+'_0'] = f

            # Compute and store indexed version of f
            f_indexed = f
            var_index = self.disc_var_list['j']
            for eqn2 in range(self.num_equations):
                for tylr in range(self.taylor_r+1):
                    E_index = self.disc_var_list[self.string_equations[eqn2][0]+str(tylr)+'_']
                    E_list = self.equation_index[self.string_equations[eqn2][0]+str(tylr)]
                    f_indexed = custom_sympy.index_subs(f_indexed,E_list,E_index,var_index,self.index_range)
            for aux in range(self.num_auxiliary):
                D_index = self.disc_var_list[self.string_auxiliary[aux][0]+'_']
                D_list = self.auxiliary_index[self.string_auxiliary[aux][0]]
                f_indexed = custom_sympy.index_subs(f_indexed,D_list,D_index,var_index,self.index_range)

            # Next lambdify the indexed version of f so it can be executed
            #   -> First, create a list of the variables in the lambdified function:
            for constant in range(self.num_constants):
                f_indexed = f_indexed.subs(self.constants_list[0][constant],self.constants_list[1][constant])
            f_dict['f'+str(eqn)+'_0_indexed'] = f_indexed
            f_dict['f'+str(eqn)+'_0_lambdified'] = sp.lambdify(lambdify_arg_list,f_indexed)

        # Compute and store derivatives of f
        for tylr in range(self.taylor_r):

            for eqn in range(self.num_equations):
                f = f_dict['f'+str(eqn)+'_'+str(tylr)].diff(self.t)
                for aux in range(self.num_auxiliary):
                    current_aux = self.auxiliary_index[self.string_auxiliary[aux][0]]
                    current_aux_deriv = self.auxiliary_index[self.string_auxiliary[aux][0]+'_deriv']
                    for idx in range(2*self.index_range+1):
                        f = f.subs(sp.Derivative(current_aux[idx],self.t),current_aux_deriv[idx])
                for eqn2 in range(self.num_equations):
                    for tylr2 in range(self.taylor_r):
                        current_var = self.equation_index[self.string_equations[eqn2][0]+str(tylr2)]
                        current_deriv = self.equation_index[self.string_equations[eqn2][0]+str(tylr2+1)]
                        for idx in range(2*self.index_range+1):
                            f = f.subs(sp.Derivative(current_var[idx],self.t),current_deriv[idx])

                f_dict['f'+str(eqn)+'_'+str(tylr+1)] = f

            # Compute and store indexed version of derivatvies of f
            var_index = self.disc_var_list['j']
            for eqn in range(self.num_equations):
                f_indexed = f_dict['f'+str(eqn)+'_'+str(tylr+1)]
                for tylr2 in range(self.taylor_r+1):
                    for eqn2 in range(self.num_equations):
                        E_index = self.disc_var_list[self.string_equations[eqn2][0]+str(tylr2)+'_']
                        E_list = self.equation_index[self.string_equations[eqn2][0]+str(tylr2)]
                        f_indexed = custom_sympy.index_subs(f_indexed,E_list,E_index,var_index,self.index_range)
                for aux in range(self.num_auxiliary):
                    D_index = self.disc_var_list[self.string_auxiliary[aux][0]+'_']
                    D_list = self.auxiliary_index[self.string_auxiliary[aux][0]]
                    f_indexed = custom_sympy.index_subs(f_indexed,D_list,D_index,var_index,self.index_range)
                for constant in range(self.num_constants):
                    f_indexed = f_indexed.subs(self.constants_list[0][constant],self.constants_list[1][constant])
                f_dict['f'+str(eqn)+'_'+str(tylr+1)+'_indexed'] = f_indexed
                f_dict['f'+str(eqn)+'_'+str(tylr+1)+'_lambdified'] = sp.lambdify(lambdify_arg_list,f_indexed)

        setattr(self,'f_dict',f_dict)

if __name__ == "__main__":
    # Sample practice script for using auto_sympy to parse an input deck.
    # It is useful to start here once you have a new input script before attempting to run using main.py
    from inputs_LTE import general_param_dict,sympy_param_dict
    from pprint import pprint
    list_equations = sympy_param_dict['list_equations']
    list_auxiliary = sympy_param_dict['list_auxiliary']
    list_spatial_vars = sympy_param_dict['list_spatial_vars']
    list_constants = sympy_param_dict['list_constants']
    temporal_var = sympy_param_dict['temporal_var']
    taylor_r = general_param_dict['taylor_r']
    index_range = general_param_dict['index_range']
    eq = equation_writer(list_equations,list_auxiliary,list_spatial_vars,list_constants,temporal_var,taylor_r,index_range)
    eq.finite_diff_approx(z=list_spatial_vars[0],dz_sym=list_constants[0][0])
    
    pprint(eq.f_dict['f0_0'])
    print(' ')
    pprint(eq.f_dict['f0_0_indexed'])
    print(' ')
    #pprint(eq.f_dict['f0_0_lambdified'])
    #print(' ')
    """
    pprint(eq.f_dict['f0_1'])
    print(' ')
    pprint(eq.f_dict['f0_1_indexed'])
    print(' ')
    #pprint(eq.f_dict['f0_1_lambdified'])
    #print(' ')
    pprint(eq.f_dict['f0_2'])
    print(' ')
    pprint(eq.f_dict['f0_2_indexed'])
    print(' ')
    #pprint(eq.f_dict['f0_2_lambdified'])
    #print(' ')
    """
    pprint(eq.f_dict['f1_0'])
    print(' ')
    pprint(eq.f_dict['f1_0_indexed'])
    print(' ')
    #pprint(eq.f_dict['f1_0_lambdified'])
    #print(' ')
    """
    pprint(eq.f_dict['f1_1'])
    print(' ')
    pprint(eq.f_dict['f1_1_indexed'])
    print(' ')
    #pprint(eq.f_dict['f1_1_lambdified'])
    #print(' ')
    pprint(eq.f_dict['f1_2'])
    print(' ')
    pprint(eq.f_dict['f1_2_indexed'])
    print(' ')
    #pprint(eq.f_dict['f1_2_lambdified'])
    #print(' ')
    #pprint(eq.d_dict)
    #pprint(eq.f_dict)

    pprint(eq.d_dict['d_0_indexed'])
    print(' ')
    #pprint(eq.d_dict['d1_indexed'])
    #print(' ')
    #pprint(eq.d_dict['d2_indexed'])
    #print(' ')

    """
