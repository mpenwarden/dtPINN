import torch
import numpy as np
import time
from scipy.special import erf

class window_sweep_class():
    def __init__(self, window_scheme, scheme_parameters, tmin, tmax):
        self.window_scheme = window_scheme
        self.scheme_parameters = scheme_parameters 
        self.end_propogation = 0 
        self.evolve_offset = tmin
        self.tmin = tmin
        self.tmax = tmax
        self.propogate_adam_iters = 0
        
        # Numerically determine the offset to add such that the scheme starts at w = 1 at t = 0
        if self.window_scheme == 'erf':
            self.sharpness = self.scheme_parameters[0]
            self.propogate_loss_tol = self.scheme_parameters[1]
            self.propogate_dt = self.scheme_parameters[2]
            self.weight_scale = self.scheme_parameters[3]
            self.tolerance = self.scheme_parameters[4]
            self.use_bc = self.scheme_parameters[5]
            self.use_null = self.scheme_parameters[6]
            
            test = np.linspace(-5,5,1000000)
            test_erf = (-abs(erf(self.sharpness*test))+1)/2
            mask = np.argwhere(np.array(test_erf) > self.tolerance).flatten()
            test = np.array(test)[mask.tolist()]
            self.method_offset = max(test) # Offset to weight scheme such that the starting t = 1
        
            self.bc_end = self.evolve_offset
            self.null_start = self.method_offset*2 + self.evolve_offset
            
        elif self.window_scheme == 'uniform':
            self.width = self.scheme_parameters[0]
            self.propogate_loss_tol = self.scheme_parameters[1]
            self.propogate_dt = self.scheme_parameters[2]
            self.weight_scale = self.scheme_parameters[3]
            self.use_bc = self.scheme_parameters[4]
            self.use_null = self.scheme_parameters[5]
            self.propogate_adam_iters = self.scheme_parameters[6]
            
            self.method_offset = self.width # Offset to weight scheme such that the starting t = 1
        
            self.bc_end = self.evolve_offset
            self.null_start = self.method_offset + self.evolve_offset
            
        elif self.window_scheme == 'linear':
            self.width = self.scheme_parameters[0]
            self.propogate_loss_tol = self.scheme_parameters[1]
            self.propogate_dt = self.scheme_parameters[2]
            self.weight_scale = self.scheme_parameters[3]
            self.use_bc = self.scheme_parameters[4]
            self.use_null = self.scheme_parameters[5]
            self.propogate_adam_iters = self.scheme_parameters[6]
            
            self.method_offset = self.width # Offset to weight scheme such that the starting t = 1
        
            self.bc_end = self.evolve_offset
            self.null_start = self.method_offset + self.evolve_offset
         
        elif self.window_scheme == 'causal':
            self.epsilon = self.scheme_parameters[0]
            self.weight_scale = self.scheme_parameters[1]
            self.use_bc = self.scheme_parameters[2]
            self.use_null = self.scheme_parameters[3]
            
            if self.use_null == 1:
                self.method_offset = 0.01 # Offset to weight scheme such that the starting t = 1
            else:
                self.method_offset = 1 # Offset to weight scheme such that the starting t = 1
        
            self.bc_end = self.evolve_offset
            self.null_start = self.method_offset + self.evolve_offset
            self.res_sum_init = 0
            self.first_prop = 1
            
    def propogate(self, evolve_offset):
        if self.propogate_adam_iters == 1:
            do_adam_iters = 1
        else: do_adam_iters = 0
                
        if self.end_propogation == 0:
            print('')
            print('Propogate')
            print('')
            self.evolve_offset += evolve_offset
                
            if self.window_scheme == 'erf':
                self.bc_end = self.evolve_offset
                self.null_start = self.method_offset*2 + self.evolve_offset
            elif self.window_scheme == 'uniform':
                self.bc_end = self.evolve_offset
                self.null_start = self.method_offset + self.evolve_offset
            elif self.window_scheme == 'linear':
                self.bc_end = self.evolve_offset
                self.null_start = self.method_offset + self.evolve_offset     
            elif self.window_scheme == 'causal':
                self.bc_end = self.evolve_offset
                self.null_start = self.method_offset + self.evolve_offset   
        
            if (self.tmax - self.bc_end - 0.0001) < 0:
                print('')
                print('End Propogate')
                print('')
                self.end_propogation = 1
        return self.end_propogation, do_adam_iters
            
    def apply_cutoffs_residualPts(self, t, x):
        if self.use_null == 1:
            real_mask = t <= torch.tensor(self.null_start)
            t_real = t[real_mask]
            x_real = x[real_mask]
        else: 
            t_real = t
            x_real = x
        
        if self.use_bc == 1:
            if self.window_scheme == 'causal' and self.use_bc == 1:
                bc_mask = t_real <= torch.tensor(self.bc_end)
                bc_mask[:100] = 0 # Keep the first set of points always on (since the weight always has to be 1 there in the causal scheme)
                t_wr = t_real[~bc_mask]
                x_wr = x_real[~bc_mask]
            else:
                bc_mask = t_real <= torch.tensor(self.bc_end)
                t_wr = t_real[~bc_mask]
                x_wr = x_real[~bc_mask]
        else:
            t_wr = t_real
            x_wr = x_real
         
        if self.use_null == 1 or self.use_bc == 1:
            return torch.unsqueeze(t_wr,-1), torch.unsqueeze(x_wr,-1)
        else:
            return t_wr, x_wr
        #return real_mask, bc_mask, torch.unsqueeze(t_wr,-1), torch.unsqueeze(x_wr,-1)
    
    def apply_cutoffs_realPts(self, t, x):
        if self.use_null == 1:
            real_mask = t <= torch.tensor(self.null_start)
            t_real = t[real_mask]
            x_real = x[real_mask]
            return torch.unsqueeze(t_real,-1), torch.unsqueeze(x_real,-1)
        else: # Without null set, all points are real
            return t, x
    
    def apply_cutoffs_bcPts(self, t, x):
        if self.use_bc == 1:
            if self.window_scheme == 'causal':
                bc_mask = t <= torch.tensor(self.bc_end)
                bc_mask[:100] = 0 # Keep the first set of points always on (since the weight always has to be 1 there in the causal scheme)
                t_bc = t[bc_mask]
                x_bc = x[bc_mask]
            else:
                bc_mask = t <= torch.tensor(self.bc_end)
                t_bc = t[bc_mask]
                x_bc = x[bc_mask]
            return torch.unsqueeze(t_bc,-1), torch.unsqueeze(x_bc,-1)
        else:
            bc_mask = t <= -1 # Should produce empty tensor
            t_bc = t[bc_mask]
            x_bc = x[bc_mask]
            return torch.unsqueeze(t_bc,-1), torch.unsqueeze(x_bc,-1)
    
    def get_weights(self, t, u, res):
        self.u_bc = []
        
        if self.window_scheme == 'erf':
            t_weight = (erf(self.sharpness*(-t.detach().numpy() + self.method_offset + self.evolve_offset))+1)/2
        elif self.window_scheme == 'uniform':
            t_weight = t.detach().numpy()*0 + 1
        elif self.window_scheme == 'linear':
            slope = (-1)/(self.width)
            t_weight = slope*t.detach().numpy()
        elif self.window_scheme == 'causal':
            #start_time = time.time() # Track training time
            t_weight = [1]
            if self.use_bc == 0:
                self.res_sum_init = 0
                
            #print(t.shape)
            #print(res.shape)
            ### For 100x100 grid sample only 
            resisual_mags = res.clone().detach()
            #print(int(len(resisual_mags)/100))
            res_t = torch.mean(torch.reshape(resisual_mags,(int(len(resisual_mags)/100),100)).square(), 1)
            res_sum = self.res_sum_init
            for i in range(int(len(resisual_mags)/100)-1):
                res_sum += res_t[i]
                t_weight.append(np.exp(-self.epsilon*res_sum))
            
            if len(t_weight) > 1:
                #print(t_weight[1])
                if self.use_bc == 1 and t_weight[1] > 0.9:# If the real step set after the first (since it always has to be 1) is above the cutoff, put the points in the BC set
                    self.bc_end = self.bc_end + 0.01
                    self.res_sum_init += res_t[1]
                    print(self.bc_end)
                    t, u = self.apply_cutoffs_bcPts(t, u)
                    if self.first_prop == 1:
                        self.u_bc = u.detach()
                        self.first_prop = 0
                    else:
                        self.u_bc = torch.cat((self.u_bc, u.detach())) # Update points saved for backward-compadability
                    
            if self.use_null == 1 and t_weight[-1] > 0.05 and self.null_start < 1.1: # If the final time set is above the cutoff, add more to the resiaul set next time
                self.null_start += 0.01
            
        return torch.unsqueeze(torch.tensor(t_weight*self.weight_scale, dtype=torch.float32), -1), self.u_bc
    
    def return_cutoffs(self):
        return self.bc_end, self.null_start