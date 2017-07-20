'''
Accuracy score counter, sigmoid and gradient descent (and normalized gradient) functions.
'''
import numpy as np


def accuracy_score(y_true, y_pred):
    '''
    Computes the accuracy of a model.
    '''
    N = len(y_true)
    count = 0
    for i in range(N):
        if y_true[i] == y_pred[i]:
            count = count+1
    return count/N


def sigmoid(s):
    return 1/(1+np.exp(-s))


def normalized_gradient(X, Y, beta, l):
    X = np.array(X)
    Y = np.array(Y)
    beta = np.array(beta)
    return np.sum([-Y[i] * X[i] * (1 - sigmoid(Y[i]*(beta.T.dot(X[i])))) for i in range(Y.shape[0])], axis=0)/Y.shape[0] + l*beta/Y.shape[0]


def gradient_descent(X, Y, epsilon=1e-6, l=1, step_size=0.1, max_steps=1000):
    beta = np.zeros(X.shape[1])
    mean = np.hstack((0, np.mean(X[:,1:], axis=0)))
    sigma = np.std(X[:,1:], axis = 0) 
    std = np.hstack((1, sigma))
    lam = np.hstack((0, l/(sigma**2)))
    X_scaled = (X-mean)/std
    for _ in range(max_steps):
        grad = normalized_gradient(X_scaled, Y, beta, lam)
        beta = beta-step_size*grad
    beta[0] = beta[0]-np.sum((mean*beta)/std)
    beta[1:] = beta[1:]/std[1:]
    return beta
