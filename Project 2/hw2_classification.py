from __future__ import division
import numpy as np
import sys

X_train = np.genfromtxt(sys.argv[1], delimiter=",")
y_train = np.genfromtxt(sys.argv[2])
X_test = np.genfromtxt(sys.argv[3], delimiter=",")

## can make more functions if required
def empirical_mean(X, y):
    # Function returning the MLE of mean for each class
    classes = np.unique(y)
    mu_hat = []
    for c in classes:
        X_extract = X[np.where(y == c)[0]]
        mu_hat.append(np.mean(X_extract, axis=0))
    return mu_hat

def empirical_covariance(X, y):
    # Function returning the MLE of covariance matrix for each class
    classes = np.unique(y)
    sigma_hat = []
    for c in classes:
        X_extract = X[np.where(y == c)[0]]
        n = np.shape(X_extract)[0]
        mu_class = np.mean(X_extract, axis=0)
        sigma_hat.append((1 / n) * np.dot(np.transpose(X_extract - mu_class), (X_extract - mu_class)))
    return sigma_hat

def empirical_prior(y):
    # Function returning the prior pi of each class
    n = np.shape(y)[0]
    classes = np.unique(y)
    pi_hat = []
    for c in classes:
        pi_hat.append(np.shape(np.where(y == c))[1] / n)
    return pi_hat

def empirical_likelihood(row, mu_y, sigma_y):
    # Function returning the likelihood of a specific row given y class distribution i.e. p(xi|yi)
    likelihood = np.linalg.det(sigma_y)**(-1/2) * np.exp((-1/2) * np.dot(np.dot((row - mu_y), np.linalg.inv(sigma_y)), np.transpose(row - mu_y)))
    return likelihood

def empirical_marginal(row, mu_hat, sigma_hat, pi_hat):
    # Function returning the marginal of a specific row i.e. p(xi) = sum over the classes(p(xi|yi))
    marginal = 0
    for j in range(len(mu_hat)):
        marginal = marginal + pi_hat[j] * empirical_likelihood(row, mu_hat[j], sigma_hat[j])
    return marginal

def pluginClassifier(X_train, y_train, X_test):    
    # this function returns the required output
    mu_hat = empirical_mean(X_train, y_train)
    sigma_hat = empirical_covariance(X_train, y_train)
    pi_hat = empirical_prior(y_train)
    final_outputs = []
    for row in X_test:
        marginal = empirical_marginal(row, mu_hat, sigma_hat, pi_hat)
        p = []
        for j in range(len(mu_hat)):
            likelihood = empirical_likelihood(row, mu_hat[j], sigma_hat[j])
            # Posterior probability that row (from X_test) belongs to calss j
            p.append(pi_hat[j] * likelihood / marginal)
        final_outputs.append(p)
    final_outputs = np.array(final_outputs)
    return final_outputs
 

final_outputs = pluginClassifier(X_train, y_train, X_test) # assuming final_outputs is returned from function

np.savetxt("probs_test.csv", final_outputs, delimiter=",") # write output to file