import numpy as np
from tensorflow import keras
from keras.layers import Dense
from keras.optimizers import Adam
from keras import  Sequential


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
    xvalid  = xv[:, x == 1]
    yvalid  = yv.reshape(num_valid)  # Solve bug   
    # Training
    model=Sequential()
    model.add(Dense(200, activation='relu', input_shape=(xtrain.shape[1],)))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=0.01),loss='binary_crossentropy')
    history = model.fit(x=xtrain, y=ytrain, epochs=5)
    
    # Prediction
    ypred   = model.predict(xvalid)
    acc     = np.sum(yvalid == ypred) / num_valid
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
