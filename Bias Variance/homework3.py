#!/usr/bin/env python3
"""
Homework 3 of Machine Learning Course

Example Usage:
    python3 homework3.py --user arsen

Make sure to test before submiting:
    python3 homework3.py test
"""


from __future__ import print_function  # If you want to run with python2
import argparse
import getpass
import matplotlib.pyplot as plt
import numpy as np


def parse_args(*argument_array):
    parser = argparse.ArgumentParser()
    parser.set_defaults(function=main)
    parser.add_argument('--user', default=getpass.getuser(),
                        help='Override system username with something else to '
                             'be include in the output file.')
    subs = parser.add_subparsers()
    test_parser = subs.add_parser('test')
    test_parser.set_defaults(function=test_function_signatures)
    args = parser.parse_args(*argument_array)
    return args


def main(args):
    """Some different values of arguments to test the function on"""

    args1 = {"num_trials": 1000, "num_points": 40, "degree": 20, "beta_star": [10, 4, -0.01, -5 / 6, 0, 1 / 24],
             "sigma": 10, "x_left_lim": -4.8, "x_right_lim": 4.8, "l": 0.01}

    args3 = {"num_trials": 1000, "num_points": 200, "degree": 20, "beta_star": [10, 4, -0.01, -5 / 6, 0, 1 / 24],
             "sigma": 10, "x_left_lim": -4.8, "x_right_lim": 4.8, "l": 0.01}

    args2 = {"num_trials": 1000, "num_points": 40, "degree": 20, "beta_star": [10, 4, -0.01, -5 / 6, 0, 1 / 24],
             "sigma": 10, "x_left_lim": -4.8, "x_right_lim": 4.8, "l": 10}

    args4 = {"num_trials": 1000, "num_points": 200, "degree": 20, "beta_star": [10, 4, -0.01, -5 / 6, 0, 1 / 24],
             "sigma": 10, "x_left_lim": -4.8, "x_right_lim": 4.8, "l": 10}

    args5 = {"num_trials": 1000, "num_points": 40, "degree": 1, "beta_star": [10, 4, -0.01, -5 / 6, 0, 1 / 24],
             "sigma": 10, "x_left_lim": -4.8, "x_right_lim": 4.8, "l": 0.01}

    args6 = {"num_trials": 1000, "num_points": 200, "degree": 1, "beta_star": [10, 4, -0.01, -5 / 6, 0, 1 / 24],
             "sigma": 10, "x_left_lim": -4.8, "x_right_lim": 4.8, "l": 0.01}

    args7 = {"num_trials": 1000, "num_points": 40, "degree": 5, "beta_star": [10, 4, -0.01, -5 / 6, 0, 1 / 24],
             "sigma": 10, "x_left_lim": -4.8, "x_right_lim": 4.8, "l": 0.01}

    args8 = {"num_trials": 1000, "num_points": 200, "degree": 5, "beta_star": [10, 4, -0.01, -5 / 6, 0, 1 / 24],
             "sigma": 10, "x_left_lim": -4.8, "x_right_lim": 4.8, "l": 0.01}

    args9 = {"num_trials": 1000, "num_points": 40, "degree": 5, "beta_star": [10, 4, -0.01, -5 / 6, 0, 1 / 24],
             "sigma": 10, "x_left_lim": -4.8, "x_right_lim": 4.8, "l": 1}

    args10 = {"num_trials": 1000, "num_points": 200, "degree": 5, "beta_star": [10, 4, -0.01, -5 / 6, 0, 1 / 24],
              "sigma": 10, "x_left_lim": -4.8, "x_right_lim": 4.8, "l": 1}

    # add more arguments

    args11 = {"num_trials": 1000, "num_points": 100, "degree": 10, "beta_star": [10, 4, -0.01, -5 / 6, 0, 1 / 24],
              "sigma": 5, "x_left_lim": -4.8, "x_right_lim": 4.8, "l": 1}

    args12 = {"num_trials": 1000, "num_points": 100, "degree": 10, "beta_star": [10, 4, -0.01, -5 / 6, 0, 1 / 24],
              "sigma": 20, "x_left_lim": -4.8, "x_right_lim": 4.8, "l": 1}


    total, bias2, var = bias_variance(**args10)

    print("total= {}, bias^2= {}, var= {}".format(total, bias2, var))


def polynomial_featurize(X, degree):
    """
    param X: a one dimensional numpy array, i.g. [x1, x2, ..., xN]
    param degree: integer - degree of the polynomial wanted
    return: a 2D np_array where row i is 1x(degree+1) np_array representing the
    polynomial. e.g. [[1, x1, x1**2, ...], [1, x2, x2**2, ...], ...]
    """
    
    X = np.array(X)
    poly = np.zeros((len(X), degree+1))
    
    for i in range(len(X)):
        for j in range(degree+1):
            poly[i][j] = X[i]**j
            
    return poly


def decompose(beta_list, beta_star):
    """
    param beta_list: a list of 1D np_arrays - beta_hats
    param beta_star: a 1D np_array
        *NOTE that beta_hat and beta_star have different dimensions
        *you need to come up with a way of comparing them
    return: a 3D tuple (total_error, bias_squared, variance)
    E[||beta_hat-beta_star||^2] = (E[beta_star - beta_hat ])^2 + E[beta_hat^2] - (E[beta_hat])^2
        *MAKE SURE you understand that each of the summations above is a number
        *UNDERSTAND that E[] is over the random variable beta_hat which depends on the random
            variable X - the data. Since we generate X many times, randomly, then beta_hat is
            a random varialbe. We are simulating its mean by trying many times.

    """
    d1 = len(beta_list[0])
    d2 = len(beta_star)
    d = max(d1, d2)
    
    beta_list = np.array(beta_list)
    beta_star = np.array(beta_star)


    
    if d1>d2:
        beta_star = np.append(beta_star, np.zeros(d1-d2))
    if d2>d1:
        new_beta_list = np.zeros((len(beta_list[1]), d2))
        for x in range(len(beta_list)):
            new_beta_list[x] = np.append(beta_list[x], np.zeros(d2-d1))
        beta_list = new_beta_list   
        
        
    # Bias squared

    sum = np.zeros(d)
    for i in range(len(beta_list)):
        sum = sum + (beta_star - beta_list[i])/d #vector
    expectation = sum #vector
    
    bias_squared = expectation.T.dot(expectation)
     
    
    # Variance
    
    sum = 0
    for i in range(len(beta_list)):
        sum = sum + beta_list[i].T.dot(beta_list[i])/d #scalar
    E1 = sum #scalar
    
    sum = np.zeros(d)
    for i in range(len(beta_list)):
        sum = sum + beta_list[i]/d #vector
    E2 = sum.T.dot(sum) #scalar
    
    variance = E1-E2
    
    # Total Error
    
    total_error = variance + bias_squared

    return total_error, bias_squar
    ed, variance




def bias_variance(num_trials, num_points, degree, beta_star, sigma, x_left_lim=0,
                  x_right_lim=100, l=1, num_red_lines=10):
    """
    param num_trials: integer - number of times we generate X given beta_star. We generate
        multiple X in order to understand the distribution of beta_hat.
    param num_points: integer - number of points in the data
    param degree: integer - degree of polynomial we use for fit_ridge_regression
    param beta_star: 1D np_array - beta_star. This is our REAL function with which poits are 
        generated. We are trying to approximate this function.
    param sigma: double - variane of normal error, when generating the point. The smaller the sigma
        is, the closer points are to the REAL function.
    param x_left_lim, x_right_lim: doubles - interval for generating x values
    param l: regularization lambda
    param num_red_lines: integer, number for red lines to show on the plot

    """
    # you don't necessarily need the below, they are just here to make the code compile
    # and give you some idea
    beta_list = []
    
    
    # For num_trials times, generate X and Y given the params above. Compute beta_hat for
    # each of them and add that to beta_list
    for _ in range(num_trials):
        X = np.random.uniform(x_left_lim, x_right_lim, num_points)
        X = polynomial_featurize(X, len(beta_star) - 1)
        Y = np.random.normal(X.dot(beta_star), sigma)
        
        beta_hat = fit_ridge_regression(X,Y,l)
        beta_list.append(beta_hat)
    
    beta_list = np.array(beta_list)


    # README.
    # We provide the code below to help you visualize bias and variance, you don't need to
    # change this.

    plt.clf()
    line_x = np.arange(x_left_lim, x_right_lim, 0.1)
    beta_list2 = beta_list[::int(num_trials / num_red_lines)]
    for beta in beta_list2:
        line_y = [polynomial_featurize([x], len(beta_star) - 1).dot(beta) for x in line_x]
        plt.plot(line_x, line_y, color='red')

    line_y = [polynomial_featurize([x], len(beta_star) - 1).dot(beta_star) for x in line_x]
    plt.plot(line_x, line_y, color='green')
    plt.scatter(X[:,1], Y)
    axes = plt.gca()
    axes.set_ylim([np.min(line_y), np.max(line_y)])
    plt.savefig("image.png", dpi=320, bbox_inches='tight')
    return decompose(beta_list, beta_star)


def fit_ridge_regression(X, Y, l=1):
    """
    :param X: data matrix (2 dimensional np.array)
    :param Y: response variables (1 dimensional np.array)
    :param l: regularization parameter lambda
    :return: value of beta (1 dimensional np.array)
    """
    X = np.array(X)
    Y = np.array(Y)
    D = X.shape[1] 
    I = np.identity(D)
    
    beta = np.linalg.inv(X.T.dot(X) + l*I).dot(X.T).dot(Y)
    
    return beta


def test_function_signatures(args):
    total, bias, variance = decompose([np.array([1, 2]), np.array([2, 3])],
                                      np.array([1., 2.5]))
    assert np.abs(total - 0.75) < 1e-10
    assert np.abs(bias - 0.25) < 1e-10
    assert np.abs(variance - 0.50) < 1e-10

    poly_vec = polynomial_featurize(np.array([1, 2]), 3)
    assert poly_vec.shape == (2, 4)
    assert list(poly_vec[1, :]) == [1, 2, 4, 8]


if __name__ == '__main__':
    args = parse_args()
    args.function(args)
