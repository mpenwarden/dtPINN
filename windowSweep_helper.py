import torch
import numpy as np
from scipy.special import erf

class window_sweep_class():
    def __init__(self, window_scheme, scheme_parameters, tmin, tmax):
        self.window_scheme = window_scheme
        self.scheme_parameters = scheme_parameters 
        self.evolve_offset = 0
        self.tmin = tmin
        self.tmax = tmax
        
        # Numerically determine the offset to add such that the scheme starts at w = 1 at t = 0
        if self.window_scheme == 'erf':
            self.sharpness = self.scheme_parameters[0]
            self.propogate_loss_tol = self.scheme_parameters[1]
            self.propogate_dt = self.scheme_parameters[2]
            self.weight_scale = self.scheme_parameters[3]
            self.tolerance = self.scheme_parameters[4]
            
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
            
            self.method_offset = self.width # Offset to weight scheme such that the starting t = 1
        
            self.bc_end = self.evolve_offset
            self.null_start = self.method_offset + self.evolve_offset
            
        elif self.window_scheme == 'linear':
            self.width = self.scheme_parameters[0]
            self.propogate_loss_tol = self.scheme_parameters[1]
            self.propogate_dt = self.scheme_parameters[2]
            self.weight_scale = self.scheme_parameters[3]
            
            self.method_offset = self.width # Offset to weight scheme such that the starting t = 1
        
            self.bc_end = self.evolve_offset
            self.null_start = self.method_offset + self.evolve_offset
            
    def propogate(self, evolve_offset):
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
            
    def apply_cutoffs_residualPts(self, t, x):
        real_mask = t < torch.tensor(self.null_start)
        t_real = t[real_mask]
        x_real = x[real_mask]
        bc_mask = t_real < torch.tensor(self.bc_end)
        t_wr = t_real[~bc_mask]
        x_wr = x_real[~bc_mask]
        return torch.unsqueeze(t_wr,-1), torch.unsqueeze(x_wr,-1)
        #return real_mask, bc_mask, torch.unsqueeze(t_wr,-1), torch.unsqueeze(x_wr,-1)
    
    def apply_cutoffs_realPts(self, t, x):
        real_mask = t < torch.tensor(self.null_start)
        t_real = t[real_mask]
        x_real = x[real_mask]
        return torch.unsqueeze(t_real,-1), torch.unsqueeze(x_real,-1)
    
    def apply_cutoffs_bcPts(self, t, x):
        bc_mask = t < torch.tensor(self.bc_end)
        t_bc = t[bc_mask]
        x_bc = x[bc_mask]
        return torch.unsqueeze(t_bc,-1), torch.unsqueeze(x_bc,-1)
    
    def get_weights(self, t):
        if self.window_scheme == 'erf':
            t_weight = (erf(self.sharpness*(-t + self.method_offset + self.evolve_offset))+1)/2
        elif self.window_scheme == 'uniform':
            t_weight = t*0 + 1
        elif self.window_scheme == 'linear':
            slope = (-1)/(self.width)
            t_weight = slope*t
            
        return torch.tensor(t_weight*self.weight_scale, dtype=torch.float32)
    
    def return_cutoffs(self):
        return self.bc_end, self.null_start