" __________________Sepehr Rezaei__________________ "
" ________________rsepehr746@gmail.com______________ "
   
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def error_rate(xtrain, ytrain, x, opts):
    
    
    fold  = opts['fold']
    xt    = fold['x_train']
    yt    = fold['y_train']
    xv    = fold['x_valid']
    yv    = fold['y_valid']
    yt=yt.values
    yv=yv.values
    # Number of instances
    num_train = np.size(xt, 0)
    num_valid = np.size(xv, 0)
    # Define selected features
    xtrain  = xt[:, x == 1]
    ytrain  = yt.reshape(num_train)  # Solve bug
    xtest  = xv[:, x == 1]
    ytest   = yv.reshape(num_valid)  # Solve bug   
    # Training
    knn=KNeighborsClassifier(n_neighbors=13)
    knn.fit(X=xtrain,y=ytrain)
    
    # Prediction
    ypred   = knn.predict(X=xtest )
    acc     = np.sum(ytest  == ypred) / num_valid
    error   = 1 - acc
    
    return error


# Error rate & Feature size
def Fun(xtrain, ytrain, x, opts):
    # Parameters
    alpha    = 0.6
    beta     = 1 - alpha
    # Original feature size
    max_feat = len(x)
    # Number of selected features
    num_feat = np.sum(x == 1)
    # Solve if no feature selected
    if num_feat == 0:
        cost  = 1
    else:
        # Get error rate
        error = error_rate(xtrain, ytrain, x, opts)
        # Objective function
        cost  = alpha * error + beta * (num_feat / max_feat)        #cost function    
        
    return cost
