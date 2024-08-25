# -*- coding: utf-8 -*-
"""This script aims to implement Radial Basis Function (RBF aka Gaussian Kernel).
Created on Thu Apr  2 21:01:29 2020

@author: Samd Guizani
Version 02

Change log:
    Version 01: Creation
    Version 02: Main program added to import data (pandas) and apply calculation"""

## Module import
import numpy as np
import pandas as pd



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
    X_rbf_kernel = np.zeros([n, n])
    for i in range(n):
        x = Xp[i]
        for j in range(n):
            y = Xp[j]
            X_rbf_kernel[i, j] = a * np.exp(- np.linalg.norm(x - y)**2 / b)
    return X_rbf_kernel


## Main program
'''Replace the file name to import the data matrix X'''
X = pd.read_csv('IRIS_data_NoHeaders_NoClass.csv', sep=',', header=None)
X = X.to_numpy()

Xp = preprocessing(X)
X_rbf = rbf_kernel(Xp, 1, 10)

# Export to csv
np.savetxt('X_rbf output.csv', X_rbf, delimiter=',')