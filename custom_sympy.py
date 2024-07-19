""" Custom SymPy functions for handling finite difference approximations and derivative substitutions.
"""

import sympy

def first_diff(func, var, delta, direction='central'):
    """ First order derivative approximation
    Args:
        func: Name of SymPy function
        var: Name of SymPy variable for the derivative approximation of the function
        delta: Cell size
        direction: Type of finite difference approximation
    Returns:
        finite difference approximation expression
    """
    if (direction == 'central'):
        return ((func.subs(var,var+delta) - func.subs(var,var-delta)) / (2*delta))
    elif (direction == 'backward'):
        return ((func - func.subs(var,var-delta)) / (delta))
    elif (direction == 'forward'):
        return ((func.subs(var,var+delta) - func) / (delta))
    elif (direction == 'backward_second_order'):
        return ((3*func - 4*func.subs(var,var-delta) + func.subs(var,var-2*delta)) / (2*delta))
    elif (direction == 'forward_second_order'):
        return ((-3*func + 4*func.subs(var,var+delta) - func.subs(var,var+2*delta)) / (2*delta))
    else:
        print("Error: not a valid direction variable")

def second_order(func, var, delta, direction='central'):
    """ Second order derivative approximation
    Args:
        func: Name of SymPy function
        var: Name of SymPy variable for the derivative approximation of the function
        delta: Cell size
        direction: Type of finite difference approximation
    Returns:
        finite difference approximation expression
    """
    if (direction == 'central'):
        return ((func.subs(var,var+delta) - 2*func + func.subs(var,var-delta)) / (delta**2))
    else:
        print("Error: not a valid direction variable")


def first_diff_sym(func, func_list, delta, direction='central'):
    """ First order derivative approximation with discrete variable renaming (f -> f_{j+1}-f_{j-1} ...)
    Args:
        func: Name of SymPy function
        func_list: List of Discretized Sympy functions: [ f_{j-3}, f_{j-2}, f_{j-1}, f_{j}, f_{j+1}, f_{j+2}, f_{j+3} ]
        delta: Cell size
        direction: Type of finite difference approximation
    Returns:
        finite difference approximation expression
    """
    if (direction == 'central'):
        return ((func_list[4] - func_list[2]) / (2*delta))
    elif (direction == 'backward'):
        return ((func_list[3] - func_list[2]) / (delta))
    elif (direction == 'forward'):
        return ((func_list[4] - func_list[3]) / (delta))
    elif (direction == 'backward_second_order'):
        return ((3*func_list[3] - 4*func_list[2] + func_list[1]) / (2*delta))
    elif (direction == 'forward_second_order'):
        return ((-3*func_list[3] + 4*func_list[4] - func_list[5]) / (2*delta))
    else:
        print("Error: not a valid direction variable")

def second_diff_sym(func, func_list, delta, direction='central'):
    """ Second order derivative approximation with discrete variable renaming (f -> f_{j+1}-f_{j-1} ...)
    Args:
        func: Name of SymPy function
        func_list: List of Discretized Sympy functions: [ f_{j-3}, f_{j-2}, f_{j-1}, f_{j}, f_{j+1}, f_{j+2}, f_{j+3} ]
        delta: Cell size
        direction: Type of finite difference approximation
    Returns:
        finite difference approximation expression
    """
    if (direction == 'central'):
        return ((func_list[4] - 2*func_list[3] + func_list[2]) / (delta**2))
    elif (direction == 'backward_second_order'):
        return ((-2*func_list[3] + 5*func_list[2] - 4*func_list[1] + func_list[0]))
    elif (direction == 'forward_second_order'):
        return ((2*func_list[3] - 5*func_list[4] + 4*func_list[5] - func_list[6]))
    else:
        print("Error: not a valid direction variable")



def list_subs_D(func, D_j_list, E_j_list, t, gamma):
    """ Returns a function func replacing d/dt[D_j] => (gamma/4)*D_j/E_j*d/dt[E_j] over the length of the D_j_list
    Args:
        func: function for substitution
        D_j_list: list of D_j terms for substitution
        E_j_list: list of E_j terms for substitution
        t: sympy time variable t
        gamma: constant parameter gamma
    Returns:
        return_func: resulting function with all the expressions replaced
    """
    return_func = func
    for i in range(len(D_j_list)):
        return_func = return_func.subs(sympy.Derivative(D_j_list[i],t),(gamma/4)*(D_j_list[i]/E_j_list[i])*sympy.Derivative(E_j_list[i],t))
    return return_func

def list_subs_E(func, E_j_list, f_j_list, t):
    """ Returns a function func replacing d/dt[E_j] => f over the length of E_j_list
    Args:
        func: function for substitution
        E_j_list: list of E_j terms for substitution
        f_j_list: list of f_j terms for substitution
        t: sympy time variable t
    Returns:
        return_func: resulting function with all the expressions replaced
    """
    return_func = func
    for i in range(len(E_j_list)):
        return_func = return_func.subs(sympy.Derivative(E_j_list[i],t),f_j_list[i])
    return return_func

def list_subs_f(func, f_j_list, dfdt_j_list, t):
    """ Returns a function func replacing d/dt[f_j] => dfdt over the length of f_j_list
    Args:
        func: function for substitution
        f_j_list: list of f_j terms for substitution
        dfdt_j_list: list of dfdt_j terms for substitution
        t: sympy time variable t
    Returns:
        return_func: resulting function with all the expressions replaced
    """
    return_func = func
    for i in range(len(f_j_list)):
        return_func = return_func.subs(sympy.Derivative(f_j_list[i],t),dfdt_j_list[i])
    return return_func

def list_subs(func, var_1_list, var_2_list):
    """ Returns a function func by replacing each element of var_1_list in the expression fun with elements from var_2_list
    Args:
        func: function for substitution
        var_1_list: list of variable for substitution
        var_2_list: list of variable for substituting for var_1
    Returns:
        return_func: resulting function with all the variables replaced
    """
    return_func = func
    for i in range(len(var_1_list)):
        return_func = return_func.subs(var_1_list[i],var_2_list[i])
    return return_func

def list_collect(func, var_list):
    """ Returns a function func with the SymPy 'collect' operation carried out for each variable in var_list
    Args:
        func: function for 'collect' operation
        var_list: list of variables for 'collect' operation
    Returns:
        return_func: resulting function with the collect operation carried out for each variable in var_list
    """
    return_func = func
    for i in range(len(var_list)):
        return_func = return_func.collect(var_list[i])
    return return_func

def index_subs(func, var_list, var_index, index, index_range):
    """ Replaces each element in the variable list with its indexed variable at the i-th index
    Args:
        func: function for substitution
        var_list: list of variable for substitution
        index_var: indexed function for substituting into func
        index: index variable for index_var
        index_range: range of index used in finite difference approximation. Ex: index_range=3 -> (j-3,...,j+3)
    Returns:
        return_func: resulting function with all the variables replaced
    """
    return_func = func
    for i in range(len(var_list)):
        return_func = return_func.subs(var_list[i],var_index[index+(i-index_range)])
    return return_func

def list_subs_func(func,expr1,expr2,expr_list):
    """ Returns a func by replacing expr1 with expr2 for each element in the lists expr_list1 and expr_list2.
    Args:
        func: function for substitution
        expr1: expression within func for substitution
        expr2: expression to substitute for expr1
        expr_list1: list of elements in expr1 for substitution
        expr_list2: list of elements in expr2 for substitution
    """
    return_func = func
    for i in range(len(expr_list)):
        return_func = return_func.subs(expr1,expr2)
        pass