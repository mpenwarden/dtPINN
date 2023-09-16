import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import torch
from pathlib import Path

# Load other helper function dependencies
from neuralNetwork_helper import Net

# Plot reference solution
def reference_plot(dataPar):
    triang_res = tri.Triangulation(dataPar['t_exact'], dataPar['x_exact']) 
    plt.figure(figsize=(10,5))
    plt.rcParams['text.usetex'] = True
    plt.tricontourf(dataPar['t_exact'], dataPar['x_exact'], dataPar['u_exact'], 100, cmap='jet')
    cbar = plt.colorbar()
    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(15)
    plt.xlabel('$t$', fontsize = 20)
    plt.ylabel('$x$', fontsize = 20)
    plt.title('Reference Solution', fontsize = 20)
    plt.show()
    
# Plot training data sampling points
def sample_plot(dataPar, userPar):
    
    plt.figure(figsize=(15,5))
    plt.rcParams['text.usetex'] = True

    plt.subplot(1,2,1) # Validate initial condition
    plt.title('Initial Condtion', fontsize = 15)
    plt.plot(dataPar['x0_decomp'].flatten(), dataPar['u0_decomp'].flatten())
    plt.xlabel('x', fontsize = 15)
    plt.ylabel('u', fontsize = 15)

    plt.subplot(1,2,2) # Validate sampling and decomposition
    plt.scatter(dataPar['t0_decomp'], dataPar['x0_decomp'], s=5) # IC
    for i in range(userPar['num_partitions']):
        plt.scatter(dataPar['tf_decomp'][i], dataPar['xf_decomp'][i], s=2) # Collocation
        plt.scatter(dataPar['tb_max_decomp'][i], dataPar['xb_max_decomp'][i], s=5) # BC
        plt.scatter(dataPar['tb_min_decomp'][i], dataPar['xb_min_decomp'][i], s=5) # BC
        if i != userPar['num_partitions']-1:
            plt.scatter(dataPar['ti_decomp'][i], dataPar['xi_decomp'][i], s=5) # Interface
    plt.xlabel('t', fontsize = 15)
    plt.ylabel('x', fontsize = 15)
    plt.title('dtPINN data', fontsize = 15)
    plt.show()

    # Paper plot of samples
    if 1 == 0: # Set manually here instead of commenting out
        plt.figure(figsize=(10,5))
        for i in range(userPar['num_partitions']):
            if i%2 == 1:
                color1 = '#1f77b4'
                color2 = '#bcbd22'
            else:
                color1 = '#ff7f0e'
                color2 = '#17becf'
            plt.scatter(dataPar['tf_decomp'][i], dataPar['xf_decomp'][i], s=1, c = color1) # Collocation
            plt.scatter(dataPar['tb_max_decomp'][i], dataPar['xb_max_decomp'][i], s=5, c = color2) # BC
            plt.scatter(dataPar['tb_min_decomp'][i], dataPar['xb_min_decomp'][i], s=5, c = color2) # BC
        for i in range(userPar['num_partitions']):
            if i != userPar['num_partitions']-1:
                plt.scatter(dataPar['ti_decomp'][i], dataPar['xi_decomp'][i], s=5, c = '#7f7f7f') # Interface
        plt.scatter(dataPar['t0_decomp'], dataPar['x0_decomp'], s=5, c = '#2ca02c') # IC
        plt.xlabel('t', fontsize = 20)
        plt.ylabel('x', fontsize = 20)
        plt.yticks(fontsize = 20); plt.xticks(fontsize = 20)
        plt.savefig(userPar['path'] + '\\log\\'+'point_set_plot', dpi = 300, bbox_inches='tight')
        
# Plot animation snapshots
def animation_plot(userPar, dataPar, model, l2_error_list, epoch_list, epoch_record_list, fig_num):
    plt.rcParams['text.usetex'] = True
    
    return_residual = 1;
    fontsize = 20; fontsize_legend = 12.5; fontsize_cbar = 15;
    
    # dtPINN
    current_num_partitions = model.return_num_partitions()
    x_decomp_test = []; x_plot_decomp_test = []; t_decomp_test = []; u_decomp_test = [];
    #decomp_num_prev = 0
    
    for i in range(userPar['num_partitions']):    
        #decomp_tuple = dataPar['sorted_decomp'][decomp_num_prev:decomp_num_prev + dataPar['decomp_num']]
        if i == 0:
            decomp_tuple = dataPar['sorted_decomp'][(dataPar['sorted_decomp'][:,0] >= dataPar['tinter'][i]) & (dataPar['sorted_decomp'][:,0] <= dataPar['tinter'][i+1])]
        else:
            decomp_tuple = dataPar['sorted_decomp'][(dataPar['sorted_decomp'][:,0] > dataPar['tinter'][i]) & (dataPar['sorted_decomp'][:,0] <= dataPar['tinter'][i+1])]
            
        t_decomp_test.append(decomp_tuple[:,0])
        x_temp = decomp_tuple[:,1]
        x_decomp_test.append(x_temp)
        x_plot_decomp_test.append(decomp_tuple[:,1])
        u_decomp_test.append(decomp_tuple[:,2])

    ### dtPINN PREDICTION
    if userPar['window_scheme'] != 'none':
        u_pred, res, t_decomp_res, x_decomp_res, u_decomp_test, t_decomp_test, x_decomp_test = model.predict(u_decomp_test, t_decomp_test, x_decomp_test, return_residual)
    else: 
        u_pred, res = model.predict(u_decomp_test, t_decomp_test, x_decomp_test, return_residual)
    
    t_decomp = np.array([]); x_decomp = np.array([]); t_res = np.array([]); x_res = np.array([]); u_test_decomp = np.array([])
    if userPar['window_scheme'] != 'none':
        for i in range(current_num_partitions):
            x_decomp = np.concatenate((x_decomp,np.array(x_decomp_test[i]).flatten()))
            t_decomp = np.concatenate((t_decomp,np.array(t_decomp_test[i]).flatten()))
            x_res = np.concatenate((x_res,np.array(x_decomp_res[i]).flatten()))
            t_res = np.concatenate((t_res,np.array(t_decomp_res[i]).flatten()))
            u_test_decomp = np.concatenate((u_test_decomp,np.array(u_decomp_test[i]).flatten()))
            
            #triang_res = tri.Triangulation(t_decomp, x_decomp)  
    else:
        x_decomp = np.array([]); t_decomp = np.array([]); u_test_decomp = np.array([])
        for i in range(current_num_partitions):
            x_decomp = np.concatenate((x_decomp,np.array(x_plot_decomp_test[i]).flatten()))
            t_decomp = np.concatenate((t_decomp,np.array(t_decomp_test[i]).flatten()))
            u_test_decomp = np.concatenate((u_test_decomp,np.array(u_decomp_test[i]).flatten()))
    
    if userPar['window_scheme'] != 'none' and len(t_res) != 0:
        triang_res = tri.Triangulation(t_res, x_res) 
        
    triang_decomp = tri.Triangulation(t_decomp, x_decomp)    
        
    u_pred_decomp = np.array([])
    res_decomp = np.array([])
    for i in range(current_num_partitions):
        u_pred_decomp = np.concatenate((u_pred_decomp, np.array(u_pred[i]).flatten()))
        if return_residual == 1:
            res_decomp = np.concatenate((res_decomp,np.array(res[i]).flatten()))
    
    cutoff_list = model.return_window_cutoffs()
    
    ### FIGURE 1
    plt.figure(figsize=(20,15))
    plt.subplot(2,2,1)
    plt.title('Exact Solution', fontsize = fontsize)
    plt.tricontourf(triang_decomp, u_test_decomp, 100 ,cmap='jet', zorder = -1)
    if userPar['window_scheme'] != 'none':
        plt.plot([cutoff_list[0][-1], cutoff_list[0][-1]], [dataPar['xmin'], dataPar['xmax']], 'k')
        plt.plot([cutoff_list[1][-1], cutoff_list[1][-1]], [dataPar['xmin'], dataPar['xmax']], 'w')
        plt.legend(['BC Set', 'Null Set'], fontsize = fontsize_legend)
    plt.xlim((dataPar['tmin'], dataPar['tmax']))
    plt.ylabel('x', fontsize = fontsize); plt.xlabel('t', fontsize = fontsize); plt.yticks(fontsize = fontsize); plt.xticks(fontsize = fontsize)
    cbar = plt.colorbar()
    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(fontsize_cbar)
        
    ### FIGURE 2
    plt.subplot(2,2,2)
    plt.title('Prediction', fontsize = fontsize)
    plt.tricontourf(triang_decomp, u_pred_decomp.flatten(), 100 ,cmap='jet')
    plt.xlim((dataPar['tmin'], dataPar['tmax']))
    plt.ylabel('x', fontsize = fontsize); plt.xlabel('t', fontsize = fontsize); plt.yticks(fontsize = fontsize); plt.xticks(fontsize = fontsize)
    cbar = plt.colorbar()
    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(fontsize_cbar)

    ### FIGURE 3
    plt.subplot(2,2,3)
    plt.title('Point-wise Error', fontsize = fontsize)
    plt.tricontourf(triang_decomp, abs(u_test_decomp-u_pred_decomp.flatten()), 100 ,cmap='jet')
    plt.xlim((dataPar['tmin'], dataPar['tmax']))
    plt.ylabel('x', fontsize = fontsize); plt.xlabel('t', fontsize = fontsize); plt.yticks(fontsize = fontsize); plt.xticks(fontsize = fontsize)
    cbar = plt.colorbar()
    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(fontsize_cbar)

    l2_error = np.linalg.norm(u_test_decomp-u_pred_decomp.flatten(), 2)/np.linalg.norm(u_test_decomp, 2)
    l2_error_list.append(l2_error)
    
    ### FIGURE 4
    plt.subplot(2,2,4)
    plt.title('Weighted Residual', fontsize = fontsize)
    if return_residual == 1:
        if userPar['window_scheme'] != 'none' and len(t_res) != 0:
            plt.tricontourf(triang_res, res_decomp.flatten(), 100 ,cmap='jet')
        else:
            plt.tricontourf(triang_decomp, res_decomp.flatten(), 100 ,cmap='jet')
    plt.xlim((dataPar['tmin'], dataPar['tmax']))
    plt.ylabel('x', fontsize = fontsize); plt.xlabel('t', fontsize = fontsize); plt.yticks(fontsize = fontsize); plt.xticks(fontsize = fontsize)
    cbar = plt.colorbar()
    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(fontsize_cbar)
    
    if userPar['save_fig'] == 1:
        plt.savefig(userPar['path'] + '\\figures\\' + userPar['model_name'] + '_fig1_' + str(fig_num), dpi = 300)
    if userPar['show_fig'] == 1:
        plt.show()
    plt.close()
    
    ### FIGURE 5
    plt.figure(figsize=(17.5,15))
    plt.subplot(2,2,1)
    plt.title('Relative L2 error', fontsize = fontsize)
    plt.plot(epoch_list, l2_error_list)
    plt.yscale('log')
    plt.ylabel('Rel L2 error', fontsize = fontsize); plt.xlabel('Epoch', fontsize = fontsize); plt.yticks(fontsize = fontsize); plt.xticks(fontsize = fontsize)
    
    ### FIGURE 6
    plt.subplot(2,2,2)
    plt.title('Loss', fontsize = fontsize)
    loss_list = model.return_loss()
    if userPar['input_encoding'] == 1:
        plt.plot(epoch_record_list, loss_list[0]); plt.plot(epoch_record_list, loss_list[1]); plt.plot(epoch_record_list, loss_list[2]); 
        plt.plot(epoch_record_list, loss_list[3]); plt.plot(epoch_record_list, loss_list[5]);
        plt.legend(['Total', 'Residual', 'Initial Condtion', 'Backward-compatibility', 'Interface'], fontsize = fontsize_legend)
    else:
        plt.plot(epoch_record_list, loss_list[0]); plt.plot(epoch_record_list, loss_list[1]); plt.plot(epoch_record_list, loss_list[2]); 
        plt.plot(epoch_record_list, loss_list[3]); plt.plot(epoch_record_list, loss_list[4]); plt.plot(epoch_record_list, loss_list[5]);
        plt.legend(['Total', 'Residual', 'Initial Condtion', 'Backward-compatibility', 'Boundary Condtion', 'Interface'], fontsize = fontsize_legend)
    plt.yscale('log')
    plt.ylabel('loss', fontsize = fontsize); plt.xlabel('Epoch', fontsize = fontsize); plt.yticks(fontsize = fontsize); plt.xticks(fontsize = fontsize)
    
    ### FIGURE 7
    plt.subplot(2,2,3)
    plt.title('Weights and bias of layers', fontsize = fontsize)
    temp = np.array([])
    box = []
    count = 0
    for name, param in model.u_dnet[current_num_partitions-1].named_parameters():
        temp = np.concatenate((temp, param.detach().numpy().flatten()))
        if count%2 == 1:
            box.append(temp)
            temp = []
        count += 1
    plt.boxplot(box);
    plt.xlabel('Layer Number', fontsize = fontsize); plt.ylabel('Weight and Bias Value', fontsize = fontsize); plt.yticks(fontsize = fontsize); plt.xticks(fontsize = fontsize)
    
    ### FIGURE 8
    plt.subplot(2,2,4)
    plt.title('Final layer outputs (at middle of current time-slab)', fontsize = fontsize)
    a = model.return_a()
    pretrained_dict = model.u_dnet[current_num_partitions-1].state_dict()
    temp_net = Net(userPar['layers'][:-1], userPar['adaptive_activation'])
    model_dict = temp_net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    temp_net.load_state_dict(pretrained_dict)
    x = np.linspace(dataPar['xmin'], dataPar['xmax'], 100)
    t = np.full(100, userPar['dt']/2 + userPar['dt']*(current_num_partitions-1))
    x = torch.unsqueeze(torch.tensor(x, dtype=torch.float32, requires_grad=True), -1)
    t = torch.unsqueeze(torch.tensor(t, dtype=torch.float32, requires_grad=True), -1)
    if userPar['input_encoding'] == 1:
        w = 2.0 * torch.tensor(np.pi) / userPar['L']
        k = torch.arange(1, userPar['M'] + 1)
        x_encode = torch.hstack([x*0+1, torch.cos(k * w * x), torch.sin(k * w * x)])
    else: 
        x_encode = 2*((x - dataPar['xmin'])/(dataPar['xmax'] - dataPar['xmin'])) - 1
    results = temp_net(torch.cat((x_encode, t), 1), a()).detach().numpy()
    for i in range(userPar['layers'][-2]):
        plt.plot(x.detach().numpy(), results[:,i])
    plt.xlabel('x', fontsize = fontsize); plt.yticks(fontsize = fontsize); plt.xticks(fontsize = fontsize)
    
    if userPar['save_fig'] == 1:
        plt.savefig(userPar['path'] + '\\figures\\' + userPar['model_name'] + '_fig2_' + str(fig_num), dpi = 300)
    if userPar['show_fig'] == 1:
        plt.show()
    plt.close()
    
    print('L2-error: ', l2_error)
    
    fig_num += 1
    return l2_error_list, fig_num

# End of training plots
def final_plot(userPar, dataPar, model, epoch_list, epoch_record_list, l2_error_list):
    plt.rcParams['text.usetex'] = True
    
    return_residual = 1
    fontsize = 20; fontsize_legend = 12.5; fontsize_cbar = 15;
    
    # dtPINN
    current_num_partitions = model.return_num_partitions()
    x_decomp_test = []; x_plot_decomp_test = []; t_decomp_test = []; u_decomp_test = [];
    #decomp_num_prev = 0
    
    for i in range(userPar['num_partitions']):    
        if i == 0:
            decomp_tuple = dataPar['sorted_decomp'][(dataPar['sorted_decomp'][:,0] >= dataPar['tinter'][i]) & (dataPar['sorted_decomp'][:,0] <= dataPar['tinter'][i+1])]
        else:
            decomp_tuple = dataPar['sorted_decomp'][(dataPar['sorted_decomp'][:,0] > dataPar['tinter'][i]) & (dataPar['sorted_decomp'][:,0] <= dataPar['tinter'][i+1])]
            
        t_decomp_test.append(decomp_tuple[:,0])
        x_temp = decomp_tuple[:,1]
        x_decomp_test.append(x_temp)
        x_plot_decomp_test.append(decomp_tuple[:,1])
        u_decomp_test.append(decomp_tuple[:,2])
        #decomp_num_prev = decomp_num_prev + dataPar['decomp_num']

    ### dtPINN PREDICTION
    if userPar['window_scheme'] != 'none':
        u_pred, res, t_decomp_res, x_decomp_res, u_decomp_test, t_decomp_test, x_decomp_test = model.predict(u_decomp_test, t_decomp_test, x_decomp_test, return_residual)
    else: 
        u_pred, res = model.predict(u_decomp_test, t_decomp_test, x_decomp_test, return_residual)
    
    t_decomp = np.array([]); x_decomp = np.array([]); t_res = np.array([]); x_res = np.array([]); u_test_decomp = np.array([])
    if userPar['window_scheme'] != 'none' and len(t_decomp_res) != 0:
        for i in range(current_num_partitions):
            x_decomp = np.concatenate((x_decomp,np.array(x_decomp_test[i]).flatten()))
            t_decomp = np.concatenate((t_decomp,np.array(t_decomp_test[i]).flatten()))
            x_res = np.concatenate((x_res,np.array(x_decomp_res[i]).flatten()))
            t_res = np.concatenate((t_res,np.array(t_decomp_res[i]).flatten()))
            u_test_decomp = np.concatenate((u_test_decomp,np.array(u_decomp_test[i]).flatten()))
            
            #triang_res = tri.Triangulation(t_decomp, x_decomp)  
    else:
        x_decomp = np.array([])
        t_decomp = np.array([])
        u_test_decomp = np.array([])
        for i in range(current_num_partitions):
            x_decomp = np.concatenate((x_decomp,np.array(x_plot_decomp_test[i]).flatten()))
            t_decomp = np.concatenate((t_decomp,np.array(t_decomp_test[i]).flatten()))
            u_test_decomp = np.concatenate((u_test_decomp,np.array(u_decomp_test[i]).flatten()))
    
    if userPar['window_scheme'] != 'none' and len(t_res) != 0:
        triang_res = tri.Triangulation(t_res, x_res) 
        
    triang_decomp = tri.Triangulation(t_decomp, x_decomp)    
        
    u_pred_decomp = np.array([])
    res_decomp = np.array([])
    for i in range(current_num_partitions):
        u_pred_decomp = np.concatenate((u_pred_decomp, np.array(u_pred[i]).flatten()))
        if return_residual == 1:
            res_decomp = np.concatenate((res_decomp,np.array(res[i]).flatten()))
    
    cutoff_list = model.return_window_cutoffs()
    
    ### FIGURE 1
    plt.figure(figsize=(10,5))
    plt.title('Exact Solution', fontsize = fontsize)
    plt.tricontourf(triang_decomp, u_test_decomp, 100 ,cmap='jet', zorder = -1)
    if userPar['window_scheme'] != 'none':
        plt.plot([cutoff_list[0][-1], cutoff_list[0][-1]], [dataPar['xmin'], dataPar['xmax']], 'k')
        plt.plot([cutoff_list[1][-1], cutoff_list[1][-1]], [dataPar['xmin'], dataPar['xmax']], 'w')
        plt.legend(['BC Set', 'Null Set'], fontsize = fontsize_legend)
    plt.xlim((dataPar['tmin'], dataPar['tmax']))
    plt.ylabel('x', fontsize = fontsize); plt.xlabel('t', fontsize = fontsize); plt.yticks(fontsize = fontsize); plt.xticks(fontsize = fontsize)
    cbar = plt.colorbar()
    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(fontsize_cbar)
    
    if userPar['save_fig'] == 1:
        plt.savefig(userPar['path'] + '\\figures\\' + userPar['model_name'] + '_FINALfig1', dpi = 300, bbox_inches='tight')
    if userPar['show_fig'] == 1:
        plt.show()
    plt.close()
    
    ### FIGURE 2
    plt.figure(figsize=(10,5))
    plt.title('Prediction', fontsize = fontsize)
    plt.tricontourf(triang_decomp, u_pred_decomp.flatten(), 100 ,cmap='jet')
    plt.xlim((dataPar['tmin'], dataPar['tmax']))
    plt.ylabel('x', fontsize = fontsize); plt.xlabel('t', fontsize = fontsize); plt.yticks(fontsize = fontsize); plt.xticks(fontsize = fontsize)
    cbar = plt.colorbar()
    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(fontsize_cbar)

    if userPar['save_fig'] == 1:
        plt.savefig(userPar['path'] + '\\figures\\' + userPar['model_name'] + '_FINALfig2', dpi = 300, bbox_inches='tight')
    if userPar['show_fig'] == 1:
        plt.show()
    plt.close()
    
    ### PREDICTION (Clean plot)
    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False; plt.rcParams['ytick.left'] = plt.rcParams['ytick.labelleft'] = False
    plt.figure(figsize=(7.5,2.5))
    plt.tricontourf(triang_decomp, u_pred_decomp.flatten(), 100 ,cmap='jet')
    plt.xlim((dataPar['tmin'], dataPar['tmax']))
    plt.xticks(fontsize = 0); plt.yticks(fontsize = 0);
    if userPar['save_fig'] == 1:
        plt.savefig(userPar['path'] + '\\figures\\' + userPar['model_name'] + '_FINALpred', dpi = 300, bbox_inches='tight')
    plt.close()
    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = True; plt.rcParams['ytick.left'] = plt.rcParams['ytick.labelleft'] = True
    
    ### FIGURE 3
    plt.figure(figsize=(10,5))
    plt.title('Point-wise Error', fontsize = fontsize)
    plt.tricontourf(triang_decomp, abs(u_test_decomp-u_pred_decomp.flatten()), 100 ,cmap='jet')
    plt.xlim((dataPar['tmin'], dataPar['tmax']))
    plt.ylabel('x', fontsize = fontsize); plt.xlabel('t', fontsize = fontsize); plt.yticks(fontsize = fontsize); plt.xticks(fontsize = fontsize)
    cbar = plt.colorbar()
    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(fontsize_cbar)

    l2_error = np.linalg.norm(u_test_decomp-u_pred_decomp.flatten(), 2)/np.linalg.norm(u_test_decomp, 2)
    
    if userPar['save_fig'] == 1:
        plt.savefig(userPar['path'] + '\\figures\\' + userPar['model_name'] + '_FINALfig3', dpi = 300, bbox_inches='tight')
    if userPar['show_fig'] == 1:
        plt.show()
    plt.close()
    
    ### FIGURE 4
    plt.figure(figsize=(10,5))
    plt.title('Weighted Residual', fontsize = fontsize)
    if return_residual == 1:
        if userPar['window_scheme'] != 'none' and len(t_res) != 0:
            plt.tricontourf(triang_res, res_decomp.flatten(), 100 ,cmap='jet')
        else:
            plt.tricontourf(triang_decomp, res_decomp.flatten(), 100 ,cmap='jet')
    plt.xlim((dataPar['tmin'], dataPar['tmax']))
    plt.ylabel('x', fontsize = fontsize); plt.xlabel('t', fontsize = fontsize); plt.yticks(fontsize = fontsize); plt.xticks(fontsize = fontsize)
    cbar = plt.colorbar()
    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(fontsize_cbar)
    
    if userPar['save_fig'] == 1:
        plt.savefig(userPar['path'] + '\\figures\\' + userPar['model_name'] + '_FINALfig4', dpi = 300, bbox_inches='tight')
    if userPar['show_fig'] == 1:
        plt.show()
    plt.close()
    
    # Not used as one may not want to compute the relative L2 error throughout training if only the final state is needed as it would add consdierable computational cost
    if userPar['animation'] == 1:
    ### FIGURE 5
        plt.figure(figsize=(10,5))
        plt.title('Relative L2 error', fontsize = fontsize)
        plt.plot(epoch_list, l2_error_list)
        plt.yscale('log')
        plt.ylabel('Rel L2 error', fontsize = fontsize); plt.xlabel('Epoch', fontsize = fontsize); plt.yticks(fontsize = fontsize); plt.xticks(fontsize = fontsize)

        if userPar['save_fig'] == 1:
            plt.savefig(userPar['path'] + '\\figures\\' + userPar['model_name'] + '_FINALfig5', dpi = 300, bbox_inches='tight')
        if userPar['show_fig'] == 1:
            plt.show()
        plt.close()
    
    ### FIGURE 6
    plt.figure(figsize=(10,5))
    plt.title('Loss', fontsize = fontsize)
    loss_list = model.return_loss()
    if userPar['input_encoding'] == 1:
        plt.plot(epoch_record_list, loss_list[0]); plt.plot(epoch_record_list, loss_list[1]); plt.plot(epoch_record_list, loss_list[2]); 
        plt.plot(epoch_record_list, loss_list[3]); plt.plot(epoch_record_list, loss_list[5]);
        plt.legend(['Total', 'Residual', 'Initial Condtion', 'Backward-compatibility', 'Interface'], fontsize = fontsize_legend)
    else:
        plt.plot(epoch_record_list, loss_list[0]); plt.plot(epoch_record_list, loss_list[1]); plt.plot(epoch_record_list, loss_list[2]); 
        plt.plot(epoch_record_list, loss_list[3]); plt.plot(epoch_record_list, loss_list[4]); plt.plot(epoch_record_list, loss_list[5]);
        plt.legend(['Total', 'Residual', 'Initial Condtion', 'Backward-compatibility', 'Boundary Condtion', 'Interface'], fontsize = fontsize_legend)
    plt.yscale('log')
    plt.ylabel('loss', fontsize = fontsize); plt.xlabel('Epoch', fontsize = fontsize); plt.yticks(fontsize = fontsize); plt.xticks(fontsize = fontsize)
    
    if userPar['save_fig'] == 1:
        plt.savefig(userPar['path'] + '\\figures\\' + userPar['model_name'] + '_FINALfig6', dpi = 300, bbox_inches='tight')
    if userPar['show_fig'] == 1:
        plt.show()
    plt.close()
    
    ### FIGURE 7
    plt.figure(figsize=(10,5))
    plt.title('Weights and bias of layers', fontsize = fontsize)
    temp = np.array([])
    box = []
    count = 0
    for name, param in model.u_dnet[current_num_partitions-1].named_parameters():
        temp = np.concatenate((temp, param.detach().numpy().flatten()))
        if count%2 == 1:
            box.append(temp)
            temp = []
        count += 1
    plt.boxplot(box);
    plt.xlabel('Layer Number', fontsize = fontsize); plt.ylabel('Weight and Bias Value', fontsize = fontsize); plt.yticks(fontsize = fontsize); plt.xticks(fontsize = fontsize)
    
    if userPar['save_fig'] == 1:
        plt.savefig(userPar['path'] + '\\figures\\' + userPar['model_name'] + '_FINALfig7', dpi = 300, bbox_inches='tight')
    if userPar['show_fig'] == 1:
        plt.show()
    plt.close()
    
    ### FIGURE 8
    plt.figure(figsize=(10,5))
    plt.title('Final layer outputs (at middle of current time-slab)', fontsize = fontsize)
    a = model.return_a()
    pretrained_dict = model.u_dnet[current_num_partitions-1].state_dict()
    temp_net = Net(userPar['layers'][:-1], userPar['adaptive_activation'])
    model_dict = temp_net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    temp_net.load_state_dict(pretrained_dict)
    x = np.linspace(dataPar['xmin'], dataPar['xmax'], 100)
    t = np.full(100, userPar['dt']/2 + userPar['dt']*(current_num_partitions-1))
    x = torch.unsqueeze(torch.tensor(x, dtype=torch.float32, requires_grad=True), -1)
    t = torch.unsqueeze(torch.tensor(t, dtype=torch.float32, requires_grad=True), -1)
    if userPar['input_encoding'] == 1:
        w = 2.0 * torch.tensor(np.pi) / userPar['L']
        k = torch.arange(1, userPar['M'] + 1)
        x_encode = torch.hstack([x*0+1, torch.cos(k * w * x), torch.sin(k * w * x)])
    else: 
        x_encode = 2*((x - dataPar['xmin'])/(dataPar['xmax'] - dataPar['xmin'])) - 1
    results = temp_net(torch.cat((x_encode, t), 1), a()).detach().numpy()
    for i in range(userPar['layers'][-2]):
        plt.plot(x.detach().numpy(), results[:,i])
    plt.xlabel('x', fontsize = fontsize); plt.yticks(fontsize = fontsize); plt.xticks(fontsize = fontsize)
    
    if userPar['save_fig'] == 1:
        plt.savefig(userPar['path'] + '\\figures\\' + userPar['model_name'] + '_FINALfig8', dpi = 300, bbox_inches='tight')
    if userPar['show_fig'] == 1:
        plt.show()
    plt.close()
    
    print('L2-error: ', l2_error)
    
    return l2_error, t_decomp, x_decomp, u_test_decomp, u_pred_decomp.flatten(), epoch_record_list, loss_list