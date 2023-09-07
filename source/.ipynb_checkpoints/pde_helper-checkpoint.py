import torch
import numpy as np
import scipy
import scipy.io

# Define PDE equation, IC, reference soltuion, etc. 
class PDE_class():
    def __init__(self, pde_type, newParams = None):
        self.pde_type = pde_type
        if self.pde_type == 'new':
            self.newParams = newParams
    
    def data(self): # Read in data from MATLAB reference solution obtained from Chebfun
        # Read in predefined example problems or newly defined problem
        if self.pde_type == 'convection':
            filename = 'data/convection.mat'
        elif self.pde_type == 'allen-cahn':
            filename = 'data/allen_cahn.mat'
        elif self.pde_type == 'heat':
            filename = 'data/heat.mat'
        elif self.pde_type == 'KS':
            filename = 'data/ks.mat'
        elif self.pde_type == 'KS_chaotic':
            filename = 'data/ks_chaotic.mat'   
        elif self.pde_type == 'kdv':
            filename = 'data/kdv.mat'
        elif self.pde_type == 'kdv_long':
            filename = 'data/kdv_long.mat'
        elif self.pde_type == 'new':
            filename = self.newParams['data']
            
        # Get solution and sample locations as well as the amount of points and bounds for future use
        data = scipy.io.loadmat(filename)
        u_exact = data['usol']
        t = data['t'][0]; x = data['x'][0]
        t_num = t.shape[0]; x_num = x.shape[0]
        T, X = np.meshgrid(t, x)
        T = T.flatten(); X = X.flatten();  u_exact = u_exact.flatten();
        tmin = T.min(); tmax = T.max(); xmin = X.min(); xmax = X.max(); 
        
        # Create data dictionary for ease of use and readability
        dataPar = {}; dataPar['u_exact'] = u_exact; dataPar['t_exact'] = T; dataPar['x_exact'] = X
        dataPar['t_num'] = t_num; dataPar['x_num'] = x_num; 
        dataPar['tmin'] = tmin; dataPar['tmax'] = tmax; dataPar['xmin'] = xmin; dataPar['xmax'] = xmax
        return dataPar
        
    def PDE(self, varibles): # Define PDE from predefined example problems or newly defined problem
        if self.pde_type == 'convection':
            u = varibles[0]; u_t = varibles[1]; u_x = varibles[2]
            eq = u_t + 30*u_x
        elif self.pde_type == 'allen-cahn':
            u = varibles[0]; u_t = varibles[1]; u_x = varibles[2]; u_xx = varibles[3]
            eq = u_t - 0.0001*u_xx+5*u*(u**2 - 1)
        elif self.pde_type == 'heat':
            u = varibles[0]; u_t = varibles[1]; u_x = varibles[2]; u_xx = varibles[3]
            eq = u_t - u_xx
        elif self.pde_type == 'KS':
            u = varibles[0]; u_t = varibles[1]; u_x = varibles[2]; u_xx = varibles[3] ; u_xxx = varibles[4] ; u_xxxx = varibles[5]
            eq = u_t + u*5*u_x + 0.5*u_xx + 0.005*u_xxxx
        elif self.pde_type == 'KS_chaotic':
            u = varibles[0]; u_t = varibles[1]; u_x = varibles[2]; u_xx = varibles[3] ; u_xxx = varibles[4] ; u_xxxx = varibles[5]
            eq = u_t + u*(100/16)*u_x + (100/16**2)*u_xx + (100/16**4)*u_xxxx
        elif self.pde_type == 'kdv' or self.pde_type == 'kdv_long':
            u = varibles[0]; u_t = varibles[1]; u_x = varibles[2]; u_xx = varibles[3] ; u_xxx = varibles[4] ;
            eq = u_t + u*u_x + 0.0025*u_xxx
        elif self.pde_type == 'new':
            for i in range(len(varibles)):
                if i == 0:
                    u = varibles[0]
                elif i == 1:
                    u_t = varibles[i]
                elif i == 2:
                    u_x = varibles[i]
                elif i == 3:
                    u_xx = varibles[i]
                elif i == 4:
                    u_xxx = varibles[i]
                elif i == 5:
                    u_xxxx = varibles[i]
            eq = eval(self.newParams['pde'])
        return eq
    
    def IC(self, x): # Define initial condtion (IC) from predefined example problems or newly defined problem
        if self.pde_type == 'convection':
            u_IC = np.sin(x)
        elif self.pde_type == 'allen-cahn':
            u_IC = x**2*np.cos(np.pi*x)
        elif self.pde_type == 'heat':
            u_IC = x*0
        elif self.pde_type == 'KS':
            u_IC = -np.sin(np.pi*x)
        elif self.pde_type == 'KS_chaotic':
            u_IC = np.cos(x)*(1+np.sin(x))
        elif self.pde_type == 'kdv' or self.pde_type == 'kdv_long':
            u_IC = np.cos(np.pi*x)
        elif self.pde_type == 'new':
            u_IC = eval(self.newParams['IC'])
            
        return np.array(u_IC).astype('float64')
    
    def bound_partials(self): # Define boundary condition partials from predefined example problems or newly defined problem
        if self.pde_type == 'convection':
            partials = []
        elif self.pde_type == 'allen-cahn':
            partials = ['u_x']
        elif self.pde_type == 'heat':
            partials = ['u_x']
        elif self.pde_type == 'KS' or self.pde_type == 'KS_chaotic':
            partials = ['u_x', 'u_xx', 'u_xxx']        
        elif self.pde_type == 'kdv' or self.pde_type == 'kdv_long':
            partials = ['u_x', 'u_xx']
        elif self.pde_type == 'new':
            partials = []
            if self.newParams['pde'].find('u_x') != -1:
                partials.append('u_x')
            if self.newParams['pde'].find('u_xx') != -1:
                partials.append('u_xx')
            if self.newParams['pde'].find('u_xxx') != -1:
                partials.append('u_xxx')
            if self.newParams['pde'].find('u_xxxx') != -1:
                partials.append('u_xxxx')
            if len(partials) > 0:
                partials = partials[:-1] # One order less than highest in the PDE
        return partials
    
    def partials(self): # Define partials needed to compute the PDE residual from predefined example problems or newly defined problem
        if self.pde_type == 'convection':
            partials = ['u_t', 'u_x']
        elif self.pde_type == 'allen-cahn':
            partials = ['u_t', 'u_x', 'u_xx']
        elif self.pde_type == 'heat':
            partials = ['u_t', 'u_x', 'u_xx']
        elif self.pde_type == 'KS' or self.pde_type == 'KS_chaotic':
            partials = ['u_t', 'u_x', 'u_xx', 'u_xxx', 'u_xxxx']
        elif self.pde_type == 'kdv' or self.pde_type == 'kdv_long':
            partials = ['u_t', 'u_x', 'u_xx', 'u_xxx']
        elif self.pde_type == 'new':
            partials = []
            if self.newParams['pde'].find('u_t') != -1: # If -1, it was not found
                partials.append('u_t')
            if self.newParams['pde'].find('u_x') != -1:
                partials.append('u_x')
            if self.newParams['pde'].find('u_xx') != -1:
                partials.append('u_xx')
            if self.newParams['pde'].find('u_xxx') != -1:
                partials.append('u_xxx')
            if self.newParams['pde'].find('u_xxxx') != -1:
                partials.append('u_xxxx')
        return partials