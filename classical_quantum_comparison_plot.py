# Plotting file for 1T Marshak wave simulation. Requires that main.py be run for both classical and quantum simulations
#   -> To switch PythonQPDE between quantum and classical configurations, change the boolean value 'quantum' in the input deck of inputs_1T.py

import matplotlib.pyplot as plt
import numpy as np

savefig = False # Save resulting figure to the output directory
showfig = True # Show the figure when running the script

parent_directory = 'output_1T/' # Output directory used in Marshak.py
classical_file = 'classical_quad_101.npz' # Place classical output file name here
quantum_file = 'quantum_quad_101.npz' # Place quantum output file name here
time_file = 'quantum_quad_101_times.npz' # Place time output file here (should be the same for either quantum or classical)
dx_file = 'quantum_quad_101_dx.npz'

with open(parent_directory+classical_file,'rb') as f:
    classical_temp_vals = np.load(f)
f.close()

with open(parent_directory+quantum_file,'rb') as f:
    quantum_temp_vals = np.load(f)
f.close()

# Times should be the same for classical and quantum. However, they will typically differ between a selected number of grid points
with open(parent_directory+time_file,'rb') as f:
    times = np.load(f)
f.close()

# Spatial grid data should be the same for classical and quantum
with open(parent_directory+dx_file,'rb') as f:
    dx = np.load(f)
f.close()
num_gridpoints = len(dx)

fig, axs = plt.subplots(1,1,sharex=True,sharey=True,figsize=(10,8))
for i in range(1,len(times)):
    axs.plot(dx,classical_temp_vals[0,:,i],label='t=%.3f Classical'%times[i])
for i in range(1,len(times)):
    axs.scatter(dx,quantum_temp_vals[0,:,i],label='t=%.3f Quantum'%times[i],s=32,marker='+',linewidth=0.75)
axs.set_title('1D Marshak - Classical and Quantum Algorithm Comparison',fontsize=20)
axs.set_ylabel('Radiation Energy Density $\epsilon$',fontsize=20)
axs.set_xlabel('Position x',fontsize=20)
axs.text(0.7,0.5,'Total Gridpoints: %i'%num_gridpoints,weight='bold',fontsize=14)
axs.legend(fontsize=12,bbox_to_anchor=(0.975,0.98))
axs.grid()
if (savefig):
    plt.savefig(parent_directory+'classical_quantum_comparison_plot.jpg')
if (showfig):
    plt.show()