import numpy as np
from keras import backend as K
import pdb

def differ(y_true, y_pred):
    return K.mean((y_true - y_pred)**2)

def eye_init(INPUT, HID):
    if HID == INPUT:
        init_weight = []
        init_weight.append(np.identity(HID))
        init_weight.append(np.zeros(HID))
        return init_weight
    else:
        u = np.floor(INPUT/HID/2)*HID    
        init_weight = []
        o = np.eye(HID,INPUT, k = u)
        init_weight.append(o.T)
        init_weight.append(np.zeros(HID))
        return init_weight


def leastsq_init(X, y, predict=None): 
    # REMOVE MEAN FROM y
    S_avg = y.mean(axis=0) 
    mean_freeY = y - S_avg

    # REMOVE MEAN FROM DATA
    try:
        predict.shape
        X_avg = predict.mean(axis=0)
        mean_freeX = predict - X_avg
    except:
        X_avg = X.mean(axis=0)
        mean_freeX = X - X_avg   
    # else:
    Cxx = np.dot(mean_freeX.T,mean_freeX)   
    Csx = np.dot(mean_freeX.T,mean_freeY)

    Wk_init = np.dot(np.linalg.pinv(Cxx), Csx)
    bk_init = S_avg-np.dot(X_avg, Wk_init)
        
    init_weight = []
    init_weight.append(Wk_init)
    init_weight.append(bk_init)
    
    
    return init_weight