# Script for converting .npz output files to .txt output files
import numpy as np

directory_name = 'output_QPDE_vs_Kull/'
num_gridpoints = 201
custom_times = np.array([0.0,0.0005,0.001,0.002,0.004,0.006,0.008,0.01,0.012])

with open(directory_name+'classical_quad_'+str(num_gridpoints)+'_dx.npz', 'rb') as f:
    dx_value = np.load(f)
f.close()

with open(directory_name+'classical_quad_'+str(num_gridpoints)+'_custom.npz', 'rb') as f:
    custom_value = np.load(f)
f.close()

# Fix end points - currently not saved in custom output
custom_value[0,0,:] = 0.15**4
custom_value[1,0,:] = 0.15
custom_value[0,-1,:] = 0.015**4
custom_value[1,-1,:] = 0.015

with open(directory_name+'dx_values.txt', 'wb') as f:
    np.savetxt(f,dx_value,fmt='%1.8f')
f.close()

with open(directory_name+'times.txt', 'wb') as f:
    np.savetxt(f,custom_times,fmt='%1.4f')
f.close()

for i in range(len(custom_value[0,0,:])):
    with open(directory_name+'rad_energy_t='+str(custom_times[i])+'.txt', 'wb') as f:
        np.savetxt(f,custom_value[0,:,i])
    f.close()
    with open(directory_name+'mat_temp_t='+str(custom_times[i])+'.txt', 'wb') as f:
        np.savetxt(f,custom_value[1,:,i])
    f.close()
    