import numpy as np
from numpy.random import rand
from function import Fun


def init_position(lb, ub, N, dim):               # return a matrix of featuers data 
    X = np.zeros([1, dim], dtype='float')
    for d in range(dim):
        X[0,d] = lb[0,d] + (ub[0,d] - lb[0,d]) * rand()        

    return X 


def binary_conversion(X, thres, N, dim):
    Xbin = np.zeros([1, dim], dtype='int')
    for d in range(dim):
        if X[0,d] > thres:
            Xbin[0,d] = 1
        else:
            Xbin[0,d] = 0

    return Xbin


def boundary(x, lb, ub):
    if x < lb:
        x = lb
    if x > ub:
        x = ub
    
    return x
 

def SCAFUN(xtrain, ytrain, opts):
    # Parameters
    ub    = 1
    lb    = 0
    thres = 0.5              #treshold
    alpha = 2       # constant
    
    max_iter  = opts['T']                  # my max iter is : 5 
    if 'alpha' in opts:                    #my alpha is : 0.1
        alpha = opts['alpha'] 
    
    # Dimension
    dim = np.size(xtrain, 1)            #my dim is : 8820
    if np.size(lb) == 1:
        ub = ub * np.ones([1, dim], dtype='float')          # my upper bound is matrix of ones [1, 8820]
        lb = lb * np.ones([1, dim], dtype='float')          # my lower bound is matrix od zeros [1, 8820]
        
    # Initialize position 
    X     = init_position(lb, ub, 1, dim)                    # return a matrix of featuers data that in my case is rand matrix [5572, 8820]
    
    # Pre
    fit   = np.zeros([1, 1], dtype='float')
    Xdb   = np.zeros([1, dim], dtype='float')
    fitD  = float('inf')
    curve = np.zeros([1, max_iter], dtype='float') 
    
    t     = 0
    while t < max_iter:
        # Binary conversion
        Xbin = binary_conversion(X, thres, 1, dim)
        
        # Fitness
        
        fit[0,0] = Fun(xtrain, ytrain, Xbin[0,:], opts)
        if fit[0,0] < fitD:            # if better vector founded update it 
            Xdb[0,:] = X[0,:]
            fitD     = fit[0,0]
        
        # Store result
        curve[0,t] = fitD.copy()
        t += 1
        
        # Parameter r1, decreases linearly from alpha to 0 (3.4)
        r1 = alpha - t * (alpha / max_iter)
        
        
        for d in range(dim):                 #  make rand number r2,r3,r4
            # Random parameter r2 & r3 & r4
            r2 = (2 * np.pi) * rand()
            r3 = 2 * rand()
            r4 = rand()
            # Position update (3.3)
            if r4 < 0.5:
                # Sine update (3.1)
                X[0,d] = X[0,d] + r1 * np.sin(r2) * abs(r3 * Xdb[0,d] - X[0,d]) 
            else:
                # Cosine update (3.2)
                X[0,d] = X[0,d] + r1 * np.cos(r2) * abs(r3 * Xdb[0,d] - X[0,d])
            
            # Boundary
            X[0,d] = boundary(X[0,d], lb[0,d], ub[0,d]) 

    
    # Best feature subset
    Gbin       = binary_conversion(Xdb, thres, 1, dim) 
    Gbin       = Gbin.reshape(dim)
    pos        = np.asarray(range(0, dim))    
    sel_index  = pos[Gbin == 1]
    num_feat   = len(sel_index)
    # Create dictionary
    sca_data = {'selected_features': sel_index, 'c': curve, 'nf': num_feat}
    
    return sca_data   
        
                    
        
        
        