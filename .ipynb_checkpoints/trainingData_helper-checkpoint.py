import torch
import numpy as np
from pyDOE import lhs

# Load other helper function dependencies
from misc_helper import lhs_sampling
from misc_helper import grid_sampling

def generate_trainData(dataPar, userPar):
    
    # Domain Settings
    tinter = [dataPar['tmin']]
    for i in range(userPar['num_partitions']-1):
        tinter.append(tinter[i] + userPar['dt'])
    tinter.append(dataPar['tmax'])
    tinter = np.array(tinter)

    N_f_decomp = int(userPar['N_f']/userPar['num_partitions'])
    N_x_decomp = int(userPar['N_x'])
    N_t_decomp = int(userPar['N_t']/userPar['num_partitions'])
    N_b_decomp = int(userPar['N_b']/userPar['num_partitions'])
    N_0_decomp = int(userPar['N_0']/1)
    
    xf_decomp = []; tf_decomp = [];
    xb_max_decomp = []; tb_max_decomp = []; 
    xb_min_decomp = []; tb_min_decomp = [];
    x0_decomp = []; t0_decomp = []; u0_decomp = []; 
    xi_decomp = []; ti_decomp = [];
    for i in range(userPar['num_partitions']):
        # Collocation
        if userPar['collocation_sampling'] == 'lhs':
            t_temp, x_temp = lhs_sampling(N_f_decomp, dataPar['xmin'], dataPar['xmax'], tinter[i], tinter[i+1])
        elif userPar['collocation_sampling'] == 'grid':
            t_temp, x_temp = grid_sampling(N_t_decomp, N_x_decomp, dataPar['xmin'], dataPar['xmax'], tinter[i], tinter[i+1])
        x_temp = [x for _, x in sorted(zip(t_temp, x_temp))]; t_temp = np.sort(t_temp);
        xf_decomp.append(x_temp); tf_decomp.append(t_temp)

        # BC
        tb_temp = np.linspace(tinter[i], tinter[i+1], N_b_decomp)
        xb_max_temp = np.full(N_b_decomp, dataPar['xmax'])
        xb_min_temp = np.full(N_b_decomp, dataPar['xmin'])
        xb_max_decomp.append(xb_max_temp); tb_max_decomp.append(tb_temp);
        xb_min_decomp.append(xb_min_temp); tb_min_decomp.append(tb_temp);

        if i != 0:
            xi_temp = np.linspace(dataPar['xmin'], dataPar['xmax'], userPar['N_i'])
            ti_temp = np.full(userPar['N_i'], tinter[i])
            xi_decomp.append(xi_temp); ti_decomp.append(ti_temp);

    # Add in points at final bound
    xi_temp = np.linspace(dataPar['xmin'], dataPar['xmax'], userPar['N_i'])
    ti_temp = np.full(userPar['N_i'], tinter[userPar['num_partitions']])
    xi_decomp.append(xi_temp); ti_decomp.append(ti_temp);

    # IC
    x0_temp = np.linspace(dataPar['xmin'], dataPar['xmax'], N_0_decomp)
    t0_temp = np.full(N_0_decomp, dataPar['tmin'])
    u0_temp = userPar['PDE'].IC(x0_temp)
    x0_decomp.append(x0_temp); t0_decomp.append(t0_temp); u0_decomp.append(u0_temp);

    xf_decomp = np.array(xf_decomp); tf_decomp = np.array(tf_decomp);
    xb_max_decomp = np.array(xb_max_decomp); tb_max_decomp = np.array(tb_max_decomp); xb_min_decomp = np.array(xb_min_decomp); tb_min_decomp = np.array(tb_min_decomp);
    x0_decomp = np.array(x0_decomp); t0_decomp = np.array(t0_decomp); u0_decomp = np.array(u0_decomp);
    xi_decomp = np.array(xi_decomp); ti_decomp = np.array(ti_decomp);
    
    dataPar['tf_decomp'] = tf_decomp; dataPar['xf_decomp'] = xf_decomp; dataPar['tb_max_decomp'] = tb_max_decomp; dataPar['xb_max_decomp'] = xb_max_decomp; 
    dataPar['tb_min_decomp'] = tb_min_decomp; dataPar['xb_min_decomp'] = xb_min_decomp; dataPar['t0_decomp'] = t0_decomp; dataPar['x0_decomp'] = x0_decomp; dataPar['u0_decomp'] = u0_decomp; 
    dataPar['ti_decomp'] = ti_decomp; dataPar['xi_decomp'] = xi_decomp;
    
    return dataPar