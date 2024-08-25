# -*- coding: utf-8 -*-
"""This script aims to implement Radial Basis Function (RBF aka Gaussian Kernel).
Created on Thu Apr  2 21:01:29 2020

@author: Samd Guizani
Version 01

Change log:
    Version 01: Creation"""

## Module import
import numpy as np


## Functions

def preprocessing(X):
    '''preprocessing(X) centers and normlizes the data by the standard deviation.
    X is a data matirx of size n rows by d columns. 
    n is the number of observations.
    d is the number of dimensions.'''
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_preproc = (X - mean) / std
    return X_preproc


def rbf_kernel(X, a, b):
    '''rbf_kernel(X) applies Radial Basis Function to a data matrix X
    X is a data matirx of size n rows by d columns. 
    n is the number of observations.
    d is the number of dimensions
    a and b are prameters such as K(x, x') = a * exp(-norm(x, x')**2 / b)'''
    n, d = np.shape(X)
    Xp = preprocessing(X)
    X_rbf_kernel = np.zeros([n, n])
    for i in range(n):
        x = Xp[i]
        for j in range(n):
            y = Xp[j]
            X_rbf_kernel[i, j] = a * np.exp(- np.linalg.norm(x - y)**2 / b)
    return X_rbf_kernel

