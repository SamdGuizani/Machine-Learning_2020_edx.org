'''This script implements Probalistic Matrix Factorization (PMF) algorithm.
Typical use is to predict missing data in a rating matix where users have rated 
objects.

Author: Samd Guizani
Date: 2020-03-28
Version: 3

Change log: 
    version 2:
        Nu, Nv definition modified
    version 3:
        Introduced function preprocess to convert train_data into matrix M (Nu x Nv)'''


from __future__ import division
import numpy as np
import sys


# Implement function here

def preprocess(train_data):
    '''Function to build the ratings matrix M of size Nu x Nv where:
        - row i is the ith user 
        - column j is the jth object. 
    train_data is matrix of size N x 3 where:
        - column 0 indexes the users (integer between 1 and Nu)
        - column 1 indexes the object 
        - column 2 indexes the rating.'''
    
    N = np.shape(train_data)[0]
    M = np.zeros([Nu, Nv])
    for idx in range(N):
        i = int(train_data[idx, 0] - 1)
        j = int(train_data[idx, 1] - 1)
        M[i, j] = train_data[idx, 2]
    return M


def V_initialize(lam, d, Nv):
    '''Initialize object location matrix V of size Nv x d'''
    V = np.zeros([Nv, d])
    for idx in range(Nv):
        V[idx] = np.random.multivariate_normal(np.zeros(d), (1/lam) * np.eye(d), 1)
    return V


def U_update(d, lam, sigma2, M, V):
    '''Update user location matrix U of size Nu x d'''
    U = np.zeros([Nu, d])
    data = M
    for idx in range(Nu):
        U[idx] = np.dot(np.linalg.inv(
                        lam * sigma2 * np.eye(d) + 
                        np.dot(np.transpose(V[np.where(data[idx] != 0)]),
                        V[np.where(data[idx] != 0)])), 
                        np.dot(data[idx], V))
    return U


def V_update(d, lam, sigma2, M, U):
    '''Update object location matrix V of size Nv x d'''
    V = np.zeros([Nv, d])
    data = np.transpose(M)
    for idx in range(Nv):
        V[idx] = np.dot(np.linalg.inv(
                        lam * sigma2 * np.eye(d) + 
                        np.dot(np.transpose(U[np.where(data[idx] != 0)]),
                        U[np.where(data[idx] != 0)])), 
                        np.dot(data[idx], U))
    return V


def obj_fun(lam, sigma2, M, U, V):
    '''Calculate the PMF objective function value'''
    error = M - np.dot(U, np.transpose(V))
    sum_error2 = np.sum(error[np.where(M !=0)]**2)
    
    U_norm = np.zeros(Nu)
    for idx in range(Nu):
        U_norm[idx] = np.dot(U[idx], np.transpose(U[idx]))
    sum_U_norm = np.sum(U_norm)
    
    V_norm = np.zeros(Nv)
    for idx in range(Nv):
        V_norm[idx] = np.dot(V[idx], np.transpose(V[idx]))
    sum_V_norm = np.sum(V_norm)
    
    obj_fun = - (1/(2 * sigma2)) * sum_error2 - (lam/2) * sum_U_norm - (lam/2) * sum_V_norm
    return obj_fun


def PMF(train_data):
    '''Implements PMF algorithm'''
    # Build the rating matrix M of size Nu x Nv. Rows = users, Columns = objects
    M = preprocess(train_data)
    
    # Initialize lists U_matrices, V_matrices, objective
    U_matrices = []
    V_matrices = []
    L = []
    
    # Initialize matrix V (Nv x d)
    V = V_initialize(lam, d, Nv)
    
    # Iterate
    for it in range(N_iterations):
        # Update matrix U (Nu x d)
        U = U_update(d, lam, sigma2, M, V)
        U_matrices.append(U)
        
        # Update matrix V (Nv x d)
        V = V_update(d, lam, sigma2, M, U)
        V_matrices.append(V)
        
        # Update objective function L
        l = obj_fun(lam, sigma2, M, U, V)
        L.append(l)
        
    return L, U_matrices, V_matrices


# Dataset import: train_data is of size N x 3
train_data = np.genfromtxt("train_data without missing ratings.csv", delimiter = ",")
# train_data = np.genfromtxt(sys.argv[1], delimiter = ",")

# Define Nu and Nv respectively the number of users and number of objects rated
Nu = len(list(set(train_data[:,0])))
Nv = len(list(set(train_data[:,1])))

# PMF parameters
lam = 2
sigma2 = 1/10
d = 5 

# Number of PMF algorithm iterations
N_iterations = 50

# Output of PMF algorithm
L, U_matrices, V_matrices = PMF(train_data)

# File saving
np.savetxt("objective.csv", L, delimiter=",")

np.savetxt("U-10.csv", U_matrices[9], delimiter=",")
np.savetxt("U-25.csv", U_matrices[24], delimiter=",")
np.savetxt("U-50.csv", U_matrices[49], delimiter=",")

np.savetxt("V-10.csv", V_matrices[9], delimiter=",")
np.savetxt("V-25.csv", V_matrices[24], delimiter=",")
np.savetxt("V-50.csv", V_matrices[49], delimiter=",")
