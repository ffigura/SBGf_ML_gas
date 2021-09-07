"""
Created on Fri Nov 13 15:42:22 2020

@author: felipe

some functions are from:

https://github.com/seg/2016-ml-contest/blob/master/LA_Team/Facies_classification_LA_TEAM_05_VALIDATION.ipynb
https://github.com/seg/2016-ml-contest/blob/master/ispl/facies_classification_try03_v2_VALIDATION.ipynb
"""

import numpy as np
import pandas as pd
import pywt
from sklearn import preprocessing

### Polynomial extension
def polynomial(X,feature_names):
    #performed without NM_M and RELPOS
    dataaux = pd.DataFrame(X[:,:-2],columns=feature_names[:-2])
    poly=preprocessing.PolynomialFeatures(degree=2).fit(dataaux)
    data_aug=poly.fit_transform(dataaux)[:,1:] #avoid Bias
    # add NM_M and RELPOS
    data_aug=np.concatenate((X[:,-2:],data_aug),axis=1)
    feature_names_aug=feature_names[-2:]+poly.get_feature_names(dataaux.columns)[1:]
    
    return(data_aug,feature_names_aug)

### Low pass filter with wavelet transform
#modified from 
#https://ataspinar.com/2018/12/21/a-guide-for-using-the-wavelet-transform-in-machine-learning/
def lowpasswav(logs, thresh = 1, wavelet="coif4"):
    filtered_signal = np.zeros((logs.shape[0], logs.shape[1]))
    for i in range(logs.shape[1]):
        thresh = thresh*np.amax(logs[:,i],axis=0)
        coeff = pywt.wavedec(logs[:,i], wavelet, mode="per" )
        coeff[1:] = (pywt.threshold(j, value=thresh, mode="soft" ) for j in coeff[1:])
        filtered_coef = pywt.waverec(coeff, wavelet, mode="per" )
        if filtered_coef.shape[0] > logs.shape[0]: #sometimes the dwt does not keep the same size
            filtered_coef = filtered_coef[:-1]
        filtered_signal[:,i] = filtered_coef
    return filtered_signal

### Gradient
def augment_features_gradient(X, depth):
    # Compute features gradient in both directions
    d_diff = np.diff(depth).reshape((-1, 1))
    d_diff[d_diff==0] = 0.001
    X_diff = np.diff(X, axis=0)
    X_grad_aux = X_diff / d_diff    
    # Compensate for last missing value
    X_grad = np.concatenate((X_grad_aux, np.zeros((1, X_grad_aux.shape[1]))))
    
    return X_grad

# Feature augmentation function - gradient and wavelet
def augment_regre(X, well, depth, feature_names):
    
    X_aug3, names_aug3 = polynomial(X,feature_names)
    
    X = np.copy(X[:,:-2])
    # Augment features
    X_aug2 = np.zeros((X.shape[0], X.shape[1]))
    for w in np.unique(well):
        w_idx = np.where(well == w)[0]
        x_aug_wav = lowpasswav(X[w_idx, :])
        X_aug2[w_idx, :] = x_aug_wav    
    
    names_aug2 = [x + " low" for x in feature_names[:-2]]

    X_aug_out=np.concatenate((X_aug3,X_aug2),axis=1)
    names_aug_out=names_aug3 + names_aug2
    
    return (X_aug_out, names_aug_out)
    
def augment_features_window(X, N_neig):
    
    # Parameters
    N_row = X.shape[0]
    N_feat = X.shape[1]

    # Zero padding
    X = np.vstack((np.zeros((N_neig, N_feat)), X, (np.zeros((N_neig, N_feat)))))

    # Loop over windows
    X_aug = np.zeros((N_row, N_feat*(2*N_neig+1)))
    for r in np.arange(N_row)+N_neig:
        this_row = []
        for c in np.arange(-N_neig,N_neig+1):
            this_row = np.hstack((this_row, X[r+c]))
        X_aug[r-N_neig] = this_row

    return X_aug


# Feature augmentation function
def augment_features(X, well, depth, N_neig=1):
    
    # Augment features
    X_aug = np.zeros((X.shape[0], X.shape[1]*(N_neig*2+2)))
    for w in np.unique(well):
        w_idx = np.where(well == w)[0]
        X_aug_win = augment_features_window(X[w_idx, :], N_neig)
        X_aug_grad = augment_features_gradient(X[w_idx, :], depth[w_idx])
        X_aug[w_idx, :] = np.concatenate((X_aug_win, X_aug_grad), axis=1)
    
    # Find padded rows
    padded_rows = np.unique(np.where(X_aug[:, 0:7] == np.zeros((1, 7)))[0])
    
    return X_aug, padded_rows    

