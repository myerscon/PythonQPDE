Python Quantum Partial Differential Equations (QPDE) solver for solving a set of coupled PDEs. The Python packages NumPy, SciPy, and SymPy are required. 
The Python package PyPDE is required to run the PyPDE_2T.py script. The current input files are for problems modeling the nonlinear radiation diffusion equations 
in one dimension. However, the input script can be modified to solve any desired set of PDEs in one-dimension (within reason).

This code is under development and subject to change and modification. This version is currently version 1.0.

Below is a brief description of the files in this directory:

# Simulation scripts:
main.py: main script for running the simulation. The input deck is specified in the 'if (__name__=="__main__"):' section by importing
            your input deck as inputs. For example, the first line of 'if (__name__=="__main__"):' is currently
            import inputs_1T as inputs
            This will use the inputs specified in inputs_1T.py as the inputs to the problem.
kacewicz_functions.py: subroutines for the integration step of the QPDE algorithm. Allows for both classical and quantum simulation modes.
auto_sympy.py: subroutines for constructing expressions for the driver functions, derivatives of the driver functions, an auxiliary functions
            as specified in the input deck. This package uses SymPy and automatically carries out analytical differentiation and discretization
            in both space and time.
custom_sympy.py: contains functions for discretizing functions continuous in both time and space into functions continuous in time yet discretized
            in space using the Finite Difference Method. These functions are used in auto_sympy.py.

# Input scripts:
inputs_1T.py: 1D radiation diffusion with material and temperature in equilibrium. Unitless.
inputs_LTE.py: 1D radiation diffusion with material and temperature not in equilibrium. This is the input script for the figures 1 and 2 in 
            "Classical-Quantum simulation of non-equilibrium Marshak waves". Unitless.
inputs_QPDE_vs_PyPDE.py: 1D radiation diffusion with material and temperature not in equilibrium. This is the input script for the figure 3 in 
            "Classical-Quantum simulation of non-equilibrium Marshak waves". Unitless.
inputs_QPDE_vs_Kull.py: 1D radiation diffusion with material and temperature not in equilibrium. This is the input script for the figures 4 and 5 in 
            "Classical-Quantum simulation of non-equilibrium Marshak waves". This is a comparison with a simulation ran using the Rad code Kull.
            This problem setup uses units.

# Other scripts:
classical_quantum_comparison_plot.py: sample script for plotting the results of main.py. It is currently configured to compare the results of a
            simulation in both classical and quantum configuration, and only plots the radiation energy density (first equation) so it can be used
            for both 1T and 2T problems.
PyPDE_2T.py: Script for running the PyPDE solver for comparison with the QPDE solver. This problem is the same as the inputs in
            inputs_QPDE_vs_PyPDE.py.
npz_to_txt.py: Script for converting .npz output files to .txt files. This can be adapted for your use by modifying the output directory and
            custom_times array to match the output of your simulation run.

# Directories:
output_final__backup/: This will appear once you run a simulation. This will contain a backup of the previous run in case output files are 
            accidentally modified or deleted.