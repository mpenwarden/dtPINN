import torch
import time
import torch.nn as nn
import numpy as np

# Load other helper function dependencies
from windowSweep_helper import window_sweep_class

# PyTorch tensor with extra dimension
def tensor(var):
    return torch.unsqueeze(torch.tensor(var, dtype=torch.float32, requires_grad=True),-1)

# Base neural network class
class Net(nn.Module):
    def __init__(self, layers, adaptive_activation, act = nn.Tanh()):
        super(Net, self).__init__()
        self.act = act; self.adaptive_activation = adaptive_activation
        self.fc = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.fc.append(nn.Linear(layers[i], layers[i+1]))
            nn.init.xavier_normal_(self.fc[-1].weight)
    
    def forward(self, x, a):
        for i in range(len(self.fc) - 1):
            x = self.fc[i](x)
            if self.adaptive_activation == 1: # Use adaptive activation functions or not
                x = self.act(a*x)
            else:
                x = self.act(x)
        x = self.fc[-1](x)
        return x
    
# Base class for adaptive activation function learnable parameter
class activation_func_a(nn.Module):
    def __init__(self, a_0):
        super(activation_func_a, self).__init__()
        self.a = nn.Parameter(torch.unsqueeze(torch.tensor(a_0, dtype=torch.float32, requires_grad=True),-1))
    def forward(self):
        return self.a
    
class weight_class(nn.Module):
    def __init__(self, w_0):
        super(weight_class, self).__init__()
        self.weight_param = nn.Parameter(torch.unsqueeze(torch.tensor(w_0, dtype=torch.float32, requires_grad=True),-1))
    def forward(self):
        return 10*(torch.tanh(self.weight_param)+1.1)

 # Main class for dtPINNs
class dtPINN:
    def __init__(self, dataPar, userPar):
        
        self.layers = userPar['layers']; self.interface_condition = userPar['interface_condition']; self.pde_type = userPar['pde_type']; self.PDE = userPar['PDE']; 
        self.dS = userPar['dS']; self.causal_dS = userPar['causal_dS']; self.layer_transfer = userPar['layer_transfer']; self.layer_trainability = userPar['layer_trainability']; 
        self.adaptive_activation = userPar['adaptive_activation']; self.input_encoding = userPar['input_encoding']; self.M = userPar['M']; self.L = userPar['L']
        self.adam_params = userPar['adam_params']; self.lbfgs_params = userPar['lbfgs_params']; self.verbose = userPar['verbose']; self.stacked_tol = userPar['stacked_tol']
        self.lambda_w = userPar['lambda_w']; self.tmin = dataPar['tmin']; self.tmax = dataPar['tmax']; self.xmin = dataPar['xmin']; self.xmax = dataPar['xmax'];
        self.dirichlet_bc = userPar['dirichlet_bc']; self.collocation_sampling = userPar['collocation_sampling']; self.userPar = userPar; self.dataPar = dataPar
        
        # Initalize window-sweeping
        self.window_scheme = userPar['window_scheme']
        if self.window_scheme != 'none':
            if userPar['num_partitions'] == 1:
                self.window_sweep = window_sweep_class(userPar['window_scheme'], userPar['scheme_parameters'], dataPar['tmin'], dataPar['tmax'])
            else:
                self.window_sweep_index = 0
                self.window_sweep = window_sweep_class(userPar['window_scheme'], userPar['scheme_parameters'], dataPar['tinter'][self.window_sweep_index], dataPar['tinter'][self.window_sweep_index+1])

            if self.window_scheme != 'causal':
                self.propogate_loss_tol = userPar['scheme_parameters'][1]
                self.propogate_dt = userPar['scheme_parameters'][2]
            else:
                self.propogate_loss_tol = 0
                self.propogate_dt = 0
                
        self.end_condition = 0 # Boolean on turning all networks back on at the end
        
        if self.causal_dS == 0: # Start with full number of dS subnetworks (i.e. for standard xPINNs where dS = n)
            self.num_partitions = self.dS # Initalize to dS
        if self.causal_dS == 1: # Add in subnetworks until dS is reached (i.e. for causal xPINNs where dS = n)
            self.num_partitions = 1 # Initalize to 1
        self.num_partitions_total = userPar['num_partitions'] # Set to gobal value
        self.learned_weights = userPar['learned_weights']
        self.prev_loss = 1; self.mse_f_prev = 1; self.first_prop = 1
        
        if self.learned_weights == 1:
            self.weights = [0, 0, 0] # [Residual Collocation,  Boundary, Interface]
        else: 
            self.weights = [-10, -10, -10] # [Residual Collocation,  Boundary, Interface]

        # Weight lists
        self.w_rc_list = []; self.w_b_list = []; self.w_i_list = [];
        # Loss lists
        self.res_loss_list = []; self.bound_loss_list = []; self.init_loss_list = []; self.bc_loss_list = []; self.inter_loss_list = []; self.weight_loss_list = []; self.total_loss_list = [];
        # Cutoff lists
        self.bc_end_list = []; self.null_start_list = [];
        
        # dtPINN weights
        self.w_rc = weight_class(self.weights[0]); self.w_b = weight_class(self.weights[1]); self.w_i = weight_class(self.weights[2])
        self.w_sum = self.w_rc().detach().numpy()[0] + self.w_b().detach().numpy()[0] + self.w_i().detach().numpy()[0]
        self.tf_decomp = tensor(dataPar['tf_decomp']); self.xf_decomp = tensor(dataPar['xf_decomp'])
        self.t0_decomp = tensor(dataPar['t0_decomp']); self.x0_decomp = tensor(dataPar['x0_decomp']); self.u0_decomp = tensor(dataPar['u0_decomp'])
        self.ti_decomp = tensor(dataPar['ti_decomp']); self.xi_decomp = tensor(dataPar['xi_decomp'])
        self.tb_max_decomp = tensor(dataPar['tb_max_decomp']); self.xb_max_decomp = tensor(dataPar['xb_max_decomp']); 
        self.tb_min_decomp = tensor(dataPar['tb_min_decomp']); self.xb_min_decomp = tensor(dataPar['xb_min_decomp']);
        if self.dirichlet_bc == 1:
            self.tb_dirichlet_max_decomp = tensor(dataPar['tb_dirichlet_max_decomp']); self.xb_dirichlet_max_decomp = tensor(dataPar['xb_dirichlet_max_decomp']); self.ub_dirichlet_max_decomp = tensor(dataPar['ub_dirichlet_max_decomp']); 
            self.tb_dirichlet_min_decomp = tensor(dataPar['tb_dirichlet_min_decomp']); self.xb_dirichlet_min_decomp = tensor(dataPar['xb_dirichlet_min_decomp']); self.ub_dirichlet_min_decomp = tensor(dataPar['ub_dirichlet_min_decomp']);
            
        ### Initalize Neural Networks
        self.u_dnet = []; self.a_list = []; self.net_params = []
        for i in range(self.num_partitions_total):
            self.u_dnet.append(Net(self.layers, self.adaptive_activation))
            self.a_list.append(activation_func_a(1))
            self.net_params += list(self.u_dnet[i].parameters()) + list(self.a_list[i].parameters())
                
        if self.learned_weights == 1:
            self.net_params += list(self.w_rc.parameters()) + list(self.w_b.parameters()) + list(self.w_i.parameters()) + list(self.w_ti.parameters())
        
        # Record Initial Loss
        bound_loss, init_loss, bc_loss, res_loss, inter_loss, weight_loss, window_vars = self.get_loss()
        loss = bound_loss + init_loss + bc_loss + res_loss + inter_loss + weight_loss
        self.append_loss(bound_loss, init_loss, bc_loss, res_loss, inter_loss, weight_loss, loss)
    
    def get_partials(self, u, t, x, bound):
        variables = [u]
        u_sum = u.sum()
        if bound == 1:
            partials_ls = self.PDE.bound_partials()
        else:
            partials_ls = self.PDE.partials()
        for i in range(len(partials_ls)):
            if partials_ls[i] == 'u_t':
                u_t = torch.autograd.grad(u_sum, t, create_graph=True)[0]
                variables.append(u_t)
            elif partials_ls[i] == 'u_x':
                u_x = torch.autograd.grad(u_sum, x, create_graph=True)[0]
                variables.append(u_x)
            elif partials_ls[i] == 'u_xx':
                u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
                variables.append(u_xx)
            elif partials_ls[i] == 'u_xxx':
                u_xxx = torch.autograd.grad(u_xx.sum(), x, create_graph=True)[0]
                variables.append(u_xxx)
            elif partials_ls[i] == 'u_xxxx':
                u_xxxx = torch.autograd.grad(u_xxx.sum(), x, create_graph=True)[0]
                variables.append(u_xxxx)
        return variables
        
    def forward(self, net, t, x, a):
        if self.input_encoding == 1:
            x = self.exact_bc_encoding(x)
        else:
            x = 2*((x - self.xmin)/(self.xmax - self.xmin))-1 # Normalize inputs to [-1,1] if there is no input encoding
        return net(torch.cat((t, x), 1), a)[:,0].reshape((-1,1))
    
    def exact_bc_encoding(self, x):
        w = 2.0 * torch.tensor(np.pi) / self.L
        k = torch.arange(1, self.M + 1)
        out = torch.hstack([x*0+1, torch.cos(k * w * x), torch.sin(k * w * x)])
        return out
    
    def interfaces(self, mse_i, u, u_prev, t, x): 
        for i in range(len(self.interface_condition)):
            if self.interface_condition[i] == 'u':
                mse_i += (u - u_prev).square().mean()

            elif self.interface_condition[i] == 'uavg':
                # Discontinous solution continuity
                u_avg = (u + u_prev)/2
                mse_i += (u_avg - u).square().mean() + (u_avg - u_prev).square().mean()
                
            elif self.interface_condition[i] == 'residual' or self.interface_condition[i] == 'rc':
                # Get partials
                variables = self.get_partials(u, t, x, 0); variables_prev = self.get_partials(u_prev, t, x, 0); 
                fu = self.PDE.PDE(variables); fu_prev = self.PDE.PDE(variables_prev)
                if self.interface_condition[i] == 'residual':
                    mse_i += fu.square().mean() + fu_prev.square().mean()
                if self.interface_condition[i] == 'rc':
                    mse_i += (fu - fu_prev).square().mean()
            
        return mse_i
    
    def get_loss(self):
        
        mse_u0 = 0; mse_ub = 0; mse_f = 0; mse_i = 0; mse_bc = 0; # Boundary, Initial, Collocation, Interface losses
        
        if (self.num_partitions - self.dS) > 0: # Compute prior interface conditions if domain that includes initial condition is not active 
            # Create Prediction for interface with the next PINN
            i = self.num_partitions - self.dS - 1
            dnet = self.u_dnet[i]; a = self.a_list[i]
            xi = self.xi_decomp[i]; ti = self.ti_decomp[i];
            ui = self.forward(dnet, ti, xi, a())
            xi_prev = xi; ti_prev = ti; ui_prev = ui;
            
        for i in range(self.num_partitions - self.dS, self.num_partitions):
            
            if i >= 0:
                ### Get index varibles
                dnet = self.u_dnet[i]; a = self.a_list[i]
                xb_max = self.xb_max_decomp[i]; tb_max = self.tb_max_decomp[i]; xb_min = self.xb_min_decomp[i]; tb_min = self.tb_min_decomp[i];
                xf = self.xf_decomp[i]; tf = self.tf_decomp[i];
                xi = self.xi_decomp[i]; ti = self.ti_decomp[i];

                ### Partition into backward-compatibility points and weighted-residual points via window-sweeping
                if self.window_scheme != 'none':
                    # Boundary points have no backward-compatibility set
                    tb_max, xb_max = self.window_sweep.apply_cutoffs_residualPts(tb_max, xb_max)
                    tb_min, xb_min = self.window_sweep.apply_cutoffs_residualPts(tb_min, xb_min)
                    # Residual points
                    tf, xf = self.window_sweep.apply_cutoffs_residualPts(tf, xf)       

                ### Boundary loss
                if self.input_encoding == 0 and self.dirichlet_bc == 0:
                    ub_max = self.forward(dnet, tb_max, xb_max, a())
                    ub_min = self.forward(dnet, tb_min, xb_min, a())
                    mse_ub += (ub_max - ub_min).square().mean()

                    # Additional terms  
                    variables_b_max = self.get_partials(ub_max, tb_max, xb_max, 1)
                    variables_b_min = self.get_partials(ub_min, tb_min, xb_min, 1)
                    for j in range(len(variables_b_max)):
                        mse_ub += (variables_b_max[j] - variables_b_min[j]).square().mean()
                
                ### Dirichlet Boundary loss
                if self.dirichlet_bc == 1:
                    ub_max = self.forward(dnet, self.tb_dirichlet_max_decomp[i], self.xb_dirichlet_max_decomp[i], a())
                    ub_min = self.forward(dnet, self.tb_dirichlet_min_decomp[i], self.xb_dirichlet_min_decomp[i], a())
                    mse_ub += (self.ub_dirichlet_max_decomp[i] - ub_max).square().mean() 
                    mse_ub += (self.ub_dirichlet_min_decomp[i] - ub_min).square().mean()
                    
                ### Initial condtion loss
                if i == 0:
                    x0 = self.x0_decomp[i]; t0 = self.t0_decomp[i]; u0 = self.u0_decomp[i];
                    # IC loss (disp)
                    u0 = self.forward(dnet, t0, x0, a())
                    mse_u0 += (self.u0_decomp - u0).square().mean()
                        
                ### Residual/collocation loss

                if len(tf) != 0:

                    uf = self.forward(dnet, tf, xf, a())
                    variables = self.get_partials(uf, tf, xf, 0)
                    fu = self.PDE.PDE(variables)
                    
                    if self.window_scheme != 'none':
                        f_weights, u_bc_causal = self.window_sweep.get_weights(tf, uf, fu) # Get weights (need fu in case causal weights are used)
                        if self.window_scheme == 'causal':
                            self.u_bc = u_bc_causal
                            fu_t = torch.unsqueeze(torch.mean(torch.reshape(fu,(int(len(fu)/100),100)).square(),1),-1)
                            mse_f += (fu_t*f_weights).mean()
                            if torch.min(f_weights) > 0.95 and int(len(fu)/100) == 100:
                                print('')
                                print('End Condition')
                                print('')
                                self.end_condition = 1
                        else:
                            mse_f += (fu*f_weights).square().mean()
                    else:
                        mse_f += fu.square().mean()
                else: uf = tf
   
                # Backward compatibility
                if self.window_scheme != 'none':
                    tf_bc, xf_bc = self.window_sweep.apply_cutoffs_bcPts(self.tf_decomp[i], self.xf_decomp[i])
                    if len(tf_bc) != 0:
                        u_bc = self.forward(dnet, tf_bc, xf_bc, a())
                        mse_bc += (self.u_bc - u_bc).square().mean()
                        
                ### Interface condtion loss    
                # Create Prediction for interface with the previous PINN
                if i != 0: 
                    ui = self.forward(dnet, ti_prev, xi_prev, a())
                    mse_i = self.interfaces(mse_i, ui, ui_prev, ti_prev, xi_prev)

                # Create Prediction for interface with the next PINN
                if i != self.num_partitions-1: # Not needed if last active network
                    ui = self.forward(dnet, ti, xi, a())
                    xi_prev = xi; ti_prev = ti; ui_prev = ui;
                
        # Sum losses
        bound_loss = self.w_b()*(mse_ub)*self.lambda_w # Boundary condition
        init_loss = self.w_b()*(mse_u0)*self.lambda_w # Initial condition
        bc_loss = self.w_b()*(mse_bc)*self.lambda_w # Backward-compadability
        res_loss = self.w_rc()*(mse_f) # Residual
        inter_loss = self.w_i()*(mse_i)*self.lambda_w # Interface
        weight_loss = (self.w_sum - (self.w_b() + self.w_rc() + self.w_i())).square().mean()
        #return bound_loss, init_loss, bc_loss, res_loss, inter_loss, weight_loss, (mse_f, uf, tf)
        return bound_loss, init_loss, bc_loss, res_loss, inter_loss, weight_loss, (bound_loss + init_loss + bc_loss + res_loss + inter_loss + weight_loss, uf, tf)
    
    def append_window_cutoffs(self):
        if self.window_scheme != 'none':
            bc_end, null_start = self.window_sweep.return_cutoffs()
            self.bc_end_list.append(bc_end); self.null_start_list.append(null_start);    
        
    def append_weights(self):
            self.w_rc_list.append(self.w_rc().detach().numpy()); self.w_b_list.append(self.w_b().detach().numpy()); 
            self.w_i_list.append(self.w_i().detach().numpy());
    
    def append_loss(self, bound_loss, init_loss, bc_loss, res_loss, inter_loss, weight_loss, loss):
            self.bound_loss_list.append(bound_loss.detach().numpy().item()); self.res_loss_list.append(res_loss.detach().numpy().item()); self.bc_loss_list.append(bc_loss.detach().numpy().item());
            self.init_loss_list.append(init_loss.detach().numpy().item()); self.inter_loss_list.append(inter_loss.detach().numpy().item());
            self.weight_loss_list.append(weight_loss.detach().numpy().item()); self.total_loss_list.append(loss.detach().numpy().item())
            
    def stacked_decomp(self):
        
        if max(self.layer_transfer) == 1: # If any layer is to be transfered, get learnable parameters
            transfer_weights = self.u_dnet[self.num_partitions-1].state_dict()
            transfer_a = self.a_list[self.num_partitions-1].state_dict()
        
        self.num_partitions += 1

        print('')
        print('New Partition')
        print('')
        
        # Initalize with previous network [Dependency on layer_transfer]
        if max(self.layer_transfer) == 1: # If any layer is to be transfered
            self.u_dnet[self.num_partitions-1].load_state_dict(transfer_weights) # Transfer weights and bias
            self.a_list[self.num_partitions-1].load_state_dict(transfer_a)
            
            i = 0
            for name, param in self.u_dnet[self.num_partitions-1].named_parameters(): # Randomize layers that are not to be transfered
                if self.layer_transfer[int(np.floor(i))] == 0:
                    param = nn.init.uniform_(torch.empty(transfer_weights[name].shape))
                i += 0.5
        
        # Turn off layers in current domain  [Dependency on layer_trainability]
        if min(self.layer_trainability) == 0: # If any layer needs to be "turned off"
            i = 0
            for name, param in self.u_dnet[self.num_partitions-1].named_parameters(): # Randomize layers that are not to be transfered
                if self.layer_trainability[int(np.floor(i))] == 0:
                    param.requires_grad = False
                i += 0.5
                
        # Turn off prior domain [Dependency on dS]
        if (self.num_partitions-(1+self.dS)) >= 0: # If the network to be turned off exists (if dS = n, it will always be a negative index and never TRUE, therefore, nothing will be turned off)
            #for name, param in self.u_dnet[self.num_partitions-2].named_parameters():
                #param.requires_grad = False
            for name, param in self.u_dnet[self.num_partitions-(1+self.dS)].named_parameters():
                param.requires_grad = False

        ### If window-sweeping is also on, reset the class with new tmin and tmax bounds
        if self.window_scheme != 'none':
            self.window_sweep_index += 1
            self.window_sweep = window_sweep_class(self.userPar['window_scheme'], self.userPar['scheme_parameters'], self.dataPar['tinter'][self.window_sweep_index], self.dataPar['tinter'][self.window_sweep_index+1])
            self.mse_f_prev = 1; self.first_prop = 1
        
        return
    
    def check_window_sweep(self, window_vars):
        mse_f = window_vars[0]; u = window_vars[1]; t = window_vars[2];
        if len(t) != 0:
            ### New code to move based on mse_f and a dt
            if (abs(self.mse_f_prev - mse_f) < torch.tensor(self.propogate_loss_tol)):

                self.end_propogate, self.do_adam_iters = self.window_sweep.propogate(self.propogate_dt)
                t, u = self.window_sweep.apply_cutoffs_bcPts(t, u)

                if self.first_prop == 1:
                    self.u_bc = u.detach()
                    self.first_prop = 0
                else:
                    self.u_bc = torch.cat((self.u_bc, u.detach())) # Update points saved for backward-compadability

            self.mse_f_prev = mse_f

            ### Old code to move based on point-wise residuals
            #mask = fu.square() < torch.tensor(self.propogate_loss_tol)
            #if len(tf[mask]) != 0:
            #    bc_end = tf[~mask][0] # Return the first value of false assuming there exists some true, this way a rogue low reisdual at the end won't cause a skip
            #    print('Check tf', tf[~mask][0])
            #    print('Check loss', fu.square()[~mask][0])
            #    self.window_sweep.propogate(bc_end.detach().numpy())
            
    def train_adam(self, optim_state, optim_state_dict, optimizer):
        self.new_partition = 0
        self.end_condition = 0
        
        if optim_state == 0:
            optimizer = torch.optim.Adam(self.net_params, lr = self.adam_params[1])
        if optim_state == 1:
            optimizer = torch.optim.Adam(self.net_params, lr = self.adam_params[1])
            optimizer.load_state_dict(optim_state_dict)
            
        for n in range(1, self.adam_params[0] + 1):
            #start_time = time.time() 
            # Get Loss 
            bound_loss, init_loss, bc_loss, res_loss, inter_loss, weight_loss, window_vars = self.get_loss()
            loss = bound_loss + init_loss + bc_loss + res_loss + inter_loss + weight_loss
                
            # Record values
            if n%100 == 0:
                self.append_loss(bound_loss, init_loss, bc_loss, res_loss, inter_loss, weight_loss, loss)
                self.append_weights(); self.append_window_cutoffs()
            # Print values
            if n%100 == 0:
                if self.verbose == 1:
                    print('epoch %d, loss: %g, delta loss: %g'%(n, loss.item(), abs(self.prev_loss-loss.item())))
            
            # Backprop and update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 
            
            if self.window_scheme != 'none':
                self.check_window_sweep(window_vars)
                #bc_end, null_start = self.window_sweep.return_cutoffs()
                #if (self.tmax - bc_end - 0.0001) < 0:
                #    print('')
                #    print('End Condition')
                #    print('')
                #    self.end_condition = 1
                #    break
                    
            if abs(self.prev_loss-loss) < self.stacked_tol and abs(self.prev_loss-loss) != 0 and loss < 1.0 and self.num_partitions != self.num_partitions_total: 
                self.new_partition = 1
                break
            
            # End training condition
            if self.window_scheme == 'none' or self.window_scheme == 'causal':
                if abs(self.prev_loss-loss) < self.stacked_tol and abs(self.prev_loss-loss) != 0 and loss < 1.0 and self.num_partitions == self.num_partitions_total and self.new_partition == 0: 
                    print('')
                    print('End Condition')
                    print('')
                    self.end_condition = 1
                    break       
            else:
                if abs(self.prev_loss-loss) < self.stacked_tol and abs(self.prev_loss-loss) != 0 and loss < 1.0 and self.num_partitions == self.num_partitions_total and self.new_partition == 0 and self.end_propogate == 1:
                    print('')
                    print('End Condition')
                    print('')
                    self.end_condition = 1
                    break
                
            self.prev_loss = loss
            
        # Check for stacked decomp condition
        if self.window_scheme == 'none':
            if self.new_partition == 1:
                finished_training = self.stacked_decomp()
        else:
            if self.new_partition == 1 and self.end_propogate == 1:
                finished_training = self.stacked_decomp()
                
        return optimizer.state_dict(), optimizer, n, self.new_partition, self.end_condition
        
    def train_lbfgs(self, optim_state, optim_state_dict, optimizer):
        global iter_count
        iter_count = 0
        self.new_partition = 0
        self.end_condition = 0
        self.do_adam_iters = 0
        
        if optim_state == 0:
            optimizer = torch.optim.LBFGS(self.net_params, lr = self.lbfgs_params[0], max_iter = self.lbfgs_params[1], max_eval = self.lbfgs_params[2],
            tolerance_grad = self.lbfgs_params[3], tolerance_change = self.lbfgs_params[4], history_size = self.lbfgs_params[5])
        if optim_state == 1:
            optimizer = torch.optim.LBFGS(self.net_params, lr = self.lbfgs_params[0], max_iter = self.lbfgs_params[1], max_eval = self.lbfgs_params[2],
            tolerance_grad = self.lbfgs_params[3], tolerance_change = self.lbfgs_params[4], history_size = self.lbfgs_params[5])
            optimizer.load_state_dict(optim_state_dict)
            
        def closure():
            global iter_count
            if self.new_partition == 0 and self.end_condition == 0 and self.do_adam_iters == 0:
                #start_time = time.time() 
                iter_count += 1
                optimizer.zero_grad()
                bound_loss, init_loss, bc_loss, res_loss, inter_loss, weight_loss, window_vars = self.get_loss()
                loss = bound_loss + init_loss + bc_loss + res_loss + inter_loss + weight_loss

                loss.backward(retain_graph=True)
                #print(time.time() - start_time)
                
                # Record values
                if iter_count%100 == 0:
                    self.append_loss(bound_loss, init_loss, bc_loss, res_loss, inter_loss, weight_loss, loss)
                    self.append_weights(); self.append_window_cutoffs()  
                # Print values
                if iter_count%100 == 0:
                    if self.verbose == 1:
                        print('epoch %d, loss: %g, delta loss: %g'%(iter_count, loss.item(), abs(self.prev_loss-loss.item())))
                
                if self.window_scheme != 'none':
                    self.check_window_sweep(window_vars)
                    #bc_end, null_start = self.window_sweep.return_cutoffs()
                    #if (self.tmax - bc_end - 0.0001) < 0:
                    #    print('')
                    #    print('End Condition')
                    #    print('')
                    #    self.end_condition = 1

                if abs(self.prev_loss-loss) < self.stacked_tol and abs(self.prev_loss-loss) != 0 and loss < 1.0 and self.num_partitions != self.num_partitions_total: 
                    self.new_partition = 1
                
                # End training condition
                if self.window_scheme == 'none' or self.window_scheme == 'causal':
                    if abs(self.prev_loss-loss) < self.stacked_tol and abs(self.prev_loss-loss) != 0 and loss < 1.0 and self.num_partitions == self.num_partitions_total and self.new_partition == 0: 
                        print('')
                        print('End Condition')
                        print('')
                        self.end_condition = 1
                else:
                    if abs(self.prev_loss-loss) < self.stacked_tol and abs(self.prev_loss-loss) != 0 and loss < 1.0 and self.num_partitions == self.num_partitions_total and self.new_partition == 0 and self.end_propogate == 1:
                        print('')
                        print('End Condition')
                        print('')
                        self.end_condition = 1
                    
                self.prev_loss = loss
                    
                return loss
            
            else: 
                return self.prev_loss
            
        optimizer.step(closure)
        
        # Check for stacked decomp condition
        # Check for stacked decomp condition
        if self.window_scheme == 'none':
            if self.new_partition == 1:
                finished_training = self.stacked_decomp()
        else:
            if self.new_partition == 1 and self.end_propogate == 1:
                finished_training = self.stacked_decomp()
            
        return optimizer.state_dict(), optimizer, iter_count, self.new_partition, self.end_condition, self.do_adam_iters
    
    def predict(self, u_decomp_test, t_decomp_test, x_decomp_test, return_residual):
        #t_decomp_test = tensor(t_decomp_test); x_decomp_test = tensor(x_decomp_test); u_decomp_test = tensor(u_decomp_test)
        
        u_out = []; res_out = []; u_test_out = []; t_test_out = []; x_test_out = []; t_res_out = []; x_res_out = []
        for i in range(self.num_partitions):
            dnet = self.u_dnet[i]
            a = self.a_list[i]
            t_test = tensor(t_decomp_test[i]);  x_test = tensor(x_decomp_test[i]); u_test = tensor(u_decomp_test[i])
            if self.window_scheme != 'none':
                t_test_temp, x_test = self.window_sweep.apply_cutoffs_realPts(t_test, x_test)
                t_test, u_test = self.window_sweep.apply_cutoffs_realPts(t_test, u_test)
                u_test_out.append(u_test.detach().numpy()); t_test_out.append(t_test.detach().numpy()); x_test_out.append(x_test.detach().numpy())
                
            u = self.forward(dnet, t_test, x_test, a())
            u_out.append(u.detach())
            if return_residual == 1:
                if self.window_scheme != 'none':
                    # Cutoff bc points
                    t_res, x_res = self.window_sweep.apply_cutoffs_residualPts(t_test, x_test)
                    if len(t_res) != 0:
                        u_res = self.forward(dnet, t_res, x_res, a())
                        variables = self.get_partials(u_res, t_res, x_res, 0)
                        res = (self.PDE.PDE(variables)).square()
                        if self.window_scheme != 'causal':
                            res_weights = self.window_sweep.get_weights(t_res, u_res, res) # Get weights (need fu in case causal weights are used)
                            res_out.append((res*res_weights).detach()); t_res_out.append(t_res.detach().numpy()); x_res_out.append(x_res.detach().numpy())
                        else:
                            res_out.append((res).detach()); t_res_out.append(t_res.detach().numpy()); x_res_out.append(x_res.detach().numpy())
                    else:
                        print('No Residual set')
                        variables = self.get_partials(u, t_test, x_test, 0)
                        res = (self.PDE.PDE(variables)).square()
                        res_out.append(res.detach())
                else:
                    variables = self.get_partials(u, t_test, x_test, 0)
                    res = (self.PDE.PDE(variables)).square()
                    res_out.append(res.detach())
                    
        if self.window_scheme != 'none':
            return u_out, res_out, t_res_out, x_res_out, u_test_out, t_test_out, x_test_out
        else: 
            return u_out, res_out
    
    def return_weights(self):
        weights = [self.w_rc_list, self.w_b_list, self.w_i_list]
        return weights
    
    def return_loss(self):
        loss = [self.total_loss_list, self.res_loss_list, self.init_loss_list, self.bc_loss_list, self.bound_loss_list, self.inter_loss_list, self.weight_loss_list]
        return loss
    
    def return_window_cutoffs(self):
        cutoffs = [self.bc_end_list, self.null_start_list]
        return cutoffs
    
    def return_interface_points(self):
        return self.xi1.detach().numpy(), self.yi1.detach().numpy(), self.fi1.detach().numpy()
    
    def return_num_partitions(self):
        return self.num_partitions
    
    def return_a(self):
        return self.a_list[self.num_partitions-1]