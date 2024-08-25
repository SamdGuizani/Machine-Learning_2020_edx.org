'''This script implements Probalistic Matrix Factorization (PMF) algorithm.
Typical use is to predict missing data in a rating matix where users have rated 
objects.

Author: Samd Guizani
Date: 2020-03-28
Version: 2
Change log: 
    version 2:
        Nu, Nv definition modified'''


from __future__ import division
import numpy as np
import sys


# Implement function here
def V_initialize(lam, d, Nv):
    '''Initialize object location matrix V of size Nv x d'''
    V = np.zeros([Nv, d])
    for idx in range(Nv):
        V[idx] = np.random.multivariate_normal(np.zeros(d), (1/lam) * np.eye(d), 1)
    return V


def U_update(d, lam, sigma2, train_data, V):
    '''Update user location matrix U of size Nu x d'''
    U = np.zeros([Nu, d])
    data = train_data
    for idx in range(Nu):
        U[idx] = np.dot(np.linalg.inv(
                        lam * sigma2 * np.eye(d) + 
                        np.dot(np.transpose(V[np.where(data[idx] != 0)]),
                        V[np.where(data[idx] != 0)])), 
                        np.dot(data[idx], V))
    return U


def V_update(d, lam, sigma2, train_data, U):
    '''Update object location matrix V of size Nv x d'''
    V = np.zeros([Nv, d])
    data = np.transpose(train_data)
    for idx in range(Nv):
        V[idx] = np.dot(np.linalg.inv(
                        lam * sigma2 * np.eye(d) + 
                        np.dot(np.transpose(U[np.where(data[idx] != 0)]),
                        U[np.where(data[idx] != 0)])), 
                        np.dot(data[idx], U))
    return V


def obj_fun(lam, sigma2, train_data, U, V):
    '''Calculate the PMF objective function value'''
    error = train_data - np.dot(U, np.transpose(V))
    sum_error2 = np.sum(error[np.where(train_data !=0)]**2)
    
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
    # Initialize lists U_matrices, V_matrices, objective
    U_matrices = []
    V_matrices = []
    L = []
    
    # Initialize matrix V (Nv x d)
    V = V_initialize(lam, d, Nv)
    
    # Iterate
    for it in range(N_iterations):
        # Update matrix U (Nu x d)
        U = U_update(d, lam, sigma2, train_data, V)
        U_matrices.append(U)
        
        # Update matrix V (Nv x d)
        V = V_update(d, lam, sigma2, train_data, U)
        V_matrices.append(V)
        
        # Update objective function L
        l = obj_fun(lam, sigma2, train_data, U, V)
        L.append(l)
        
    return L, U_matrices, V_matrices


# Assuming the PMF function returns Loss L, U_matrices and V_matrices (refer to lecture)

train_data = np.genfromtxt("PMF_Test_DataSet.csv", delimiter = ",")
Nu, Nv = np.shape(train_data)

'''train_data = np.genfromtxt(sys.argv[1], delimiter = ",")
Nu = len(list(set(train_data[:,0])))
Nv = len(list(set(train_data[:,1])))'''

lam = 2
sigma2 = 1/10
d = 5 
N_iterations = 50

L, U_matrices, V_matrices = PMF(train_data)

np.savetxt("objective.csv", L, delimiter=",")

np.savetxt("U-10.csv", U_matrices[9], delimiter=",")
np.savetxt("U-25.csv", U_matrices[24], delimiter=",")
np.savetxt("U-50.csv", U_matrices[49], delimiter=",")

np.savetxt("V-10.csv", V_matrices[9], delimiter=",")
np.savetxt("V-25.csv", V_matrices[24], delimiter=",")
np.savetxt("V-50.csv", V_matrices[49], delimiter=",")
