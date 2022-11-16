import torch
import numpy as np
from pyDOE import lhs
import time
import imageio
import glob
from pathlib import Path

# Load other helper function dependencies
from plot_helper import result_plot

# Latin-hypercube sampling
def lhs_sampling(points, xmin, xmax, tmin, tmax):
    loc = lhs(2, samples = points) 
    x = xmin + (xmax-xmin)*loc[:,0]; t = tmin + (tmax-tmin)*loc[:,1]
    return t, x

# Grid sampling
def grid_sampling(points_t, points_x, xmin, xmax, tmin, tmax):
    t = np.linspace(tmin, tmax, points_t)
    x = np.linspace(xmin, xmax, points_x)
    T, X = np.meshgrid(t, x)
    return T.flatten(), X.flatten()

def combine_layers(input_encoding, M, hidden_layers):
    if input_encoding == 1:
        layers = [M*2 + 2] + hidden_layers + [1]; # PINN layers
    else:
        layers = [2] + hidden_layers + [1]; # PINN layers
    return layers

def get_dt(tmin, tmax, num_partitions):
    return (abs(tmin-tmax))/num_partitions

def get_userPar(pde_type, input_encoding, M, L, layers, adam_params, lbfgs_params, adaptive_activation, learned_weights, N_x, N_t, N_f, N_0, N_b, N_i, collocation_sampling,\
                     num_partitions, dt, dS, causal_dS, interface_condition, layer_transfer, layer_trainability, window_scheme, scheme_parameters, PDE, stacked_tol):
    userPar = {}; userPar['pde_type'] = pde_type;
    userPar['input_encoding'] = input_encoding; userPar['M'] = M;  userPar['L'] = L; userPar['layers'] = layers;  userPar['adam_params'] = adam_params; 
    userPar['lbfgs_params'] = lbfgs_params; userPar['adaptive_activation'] = adaptive_activation; userPar['learned_weights'] = learned_weights; 
    userPar['N_x'] = N_x; userPar['N_t'] = N_t; userPar['N_f'] = N_f; userPar['N_0'] = N_0; userPar['N_b'] = N_b; userPar['N_i'] = N_i; 
    userPar['collocation_sampling'] = collocation_sampling; userPar['num_partitions'] = num_partitions; userPar['dt'] = dt; userPar['dS'] = dS; userPar['causal_dS'] = causal_dS;
    userPar['interface_condition'] = interface_condition; userPar['layer_transfer'] = layer_transfer; userPar['layer_trainability'] = layer_trainability; 
    userPar['window_scheme'] = window_scheme; userPar['scheme_parameters'] = scheme_parameters; userPar['PDE'] = PDE; userPar['stacked_tol'] = stacked_tol;
    return userPar

def make_gif(model_name):
    impath1 = glob.glob(str(Path.cwd()) + '\\figures\\' + model_name + '_fig1_*.png')
    impath2 = glob.glob(str(Path.cwd()) + '\\figures\\' + model_name + '_fig2_*.png')
    images1 = []; images2 = []
    for filename in impath1:
        images1.append(imageio.imread(filename))
    for filename in impath2:
        images2.append(imageio.imread(filename))
    imageio.mimsave(str(Path.cwd()) + '\\figures\\' + model_name + '_1.mp4', images1, macro_block_size = 1)
    imageio.mimsave(str(Path.cwd()) + '\\figures\\' + model_name + '_2.mp4', images2, macro_block_size = 1)
    
def training_loop(userPar, dataPar, model, adam_loops, lbfgs_loops):
    
    l2_error_list = [1]; epoch_list = [0]; epoch_record_list = [0]; fig_num = 0; iter_count = 0

    dataPar['decomp_num'] = dataPar['x_num']*int(dataPar['t_num']/userPar['num_partitions'])
    dataPar['sorted_decomp'] = np.array(sorted(zip(dataPar['t_exact'], dataPar['x_exact'], dataPar['u_exact'])))

    print('dtPINN'); train_time = 0

    ### Adam training
    optim_state_dict = []
    optimizer = 0
    optim_state = 0
    for i in range(adam_loops):
        print('Train Adam:')
        start_time = time.time() # Track training time
        optim_state_dict, optimizer, iterations, new_partition, end_condition = model.train_adam(optim_state, optim_state_dict, optimizer)
        elapsed = time.time() - start_time
        train_time += elapsed
        print('Training time: %.4f' % (train_time))
        #print('Iterations', iterations)
        optim_state = 1
        # Plot
        if userPar['save_fig'] == 1 or userPar['show_fig'] == 1:
            l2_error_list, epoch_list, epoch_record_list, fig_num = result_plot(userPar, dataPar, model, iterations, l2_error_list, epoch_list, epoch_record_list, fig_num)

        if new_partition == 1: # If new partition was added, reset the optimization state
            optim_state = 0
        if end_condition == 1: # If end condition triggered, break out of training loop
            break
            
    ### L-BFGS training
    optim_state_dict = []
    optimizer = 0
    optim_state = 0
    for i in range(lbfgs_loops):
        print('Train L-BFGS:')
        start_time = time.time() # Track training time
        optim_state_dict, optimizer, iterations, new_partition, end_condition = model.train_lbfgs(optim_state, optim_state_dict, optimizer)
        elapsed = time.time() - start_time        
        train_time += elapsed
        print('Training time: %.4f' % (train_time))
        #print('Iterations', iterations)
        optim_state = 1
        # Plot
        if userPar['save_fig'] == 1 or userPar['show_fig'] == 1:
            l2_error_list, epoch_list, epoch_record_list, fig_num = result_plot(userPar, dataPar, model, iterations, l2_error_list, epoch_list, epoch_record_list, fig_num)

        if new_partition == 1: # If new partition was added, reset the optimization state
            optim_state = 0
        if end_condition == 1: # If end condition triggered, break out of training loop
            break
        
    return model