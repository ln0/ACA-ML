'''
File to compile DT, RF, LR
'''

import numpy as np
from random_forest import RandomForest
from decision_tree import DecisionTree
from gradient_descent import accuracy_score, sigmoid, gradient_descent

def evaluate_performance():
    '''
    Evaluate the performance of decision trees and logistic regression,
    average over 1,000 trials of 10-folds cross validation
    Return:
      a matrix giving the performance that will contain the following entries:
      stats[0,0] = mean accuracy of decision tree
      stats[0,1] = std deviation of decision tree accuracy
      stats[1,0] = mean accuracy of logistic regression
      stats[1,1] = std deviation of logistic regression accuracy
    ** Note that your implementation must follow this API**
    '''

    folds = 10
    data = np.loadtxt('SPECTF.dat', delimiter=',')
    X = np.array(data[:, 1:])
    Y = np.array([data[:, 0]]).T
    n = X.shape[0]
    
    dt_accuracies = []
    rf_accuracies = []
    log_accuracies = []
    
    for trial in range(10):
        idx = np.arange(n)
        #np.random.seed(13)
        np.random.shuffle(idx)

        X = X[idx]
        Y = Y[idx]

        tree_acc = []
        forest_acc = []
        log_acc = []
        for it in range(folds):
            X_test = X[it::folds, :]
            Y_test = Y[it::folds, :]
    
            X_train = [X[i] for i in range(len(X)) if i % folds != it]
            Y_train = [Y[i] for i in range(len(Y)) if i % folds != it]

            
            # Decision Tree Classifier
            classifier_dt = DecisionTree(15)
            classifier_dt.fit(X_train, Y_train)
            Y_pred = classifier_dt.predict(X_test)
            tree_acc.append(accuracy_score(Y_test, Y_pred))


            # Random Forest Classifier
            classifier_rf = RandomForest(40, 15)
            classifier_rf.fit(X_train, Y_train)
            Y_pred = classifier_rf.predict(X_test)[0]
            forest_acc.append(accuracy_score(Y_test, Y_pred))


            # Logistic Regression Classifier
            X_train = np.array(X_train)
            X_train = np.column_stack((np.ones(len(X_train)), X_train))
            Y_train = [1 if label == 1 else -1 for label in Y_train]
            beta = gradient_descent(X_train, Y_train, epsilon=1e-3, l=1, step_size=0.1, max_steps=200)
            Y_pred = [1 if label >= 0 else -1 for label in X_train.dot(beta)]
            log_acc.append(accuracy_score(Y_train, Y_pred))


        dt_accuracies.append(np.mean(tree_acc))
        rf_accuracies.append(np.mean(forest_acc))
        log_accuracies.append(np.mean(log_acc))

    # compute the training accuracy of the models
    meanDecisionTreeAccuracy = np.mean(dt_accuracies)
    stddevDecisionTreeAccuracy = np.std(dt_accuracies)
    
    meanRandomForestAccuracy = np.mean(rf_accuracies)
    stddevRandomForestAccuracy = np.std(rf_accuracies)
    
    meanLogisticRegressionAccuracy = np.mean(log_accuracies)
    stddevLogisticRegressionAccuracy = np.std(log_accuracies)


    stats = np.zeros((3, 2))
    stats[0, 0] = meanDecisionTreeAccuracy
    stats[0, 1] = stddevDecisionTreeAccuracy
    stats[1, 0] = meanRandomForestAccuracy
    stats[1, 1] = stddevRandomForestAccuracy
    stats[2, 0] = meanLogisticRegressionAccuracy
    stats[2, 1] = stddevLogisticRegressionAccuracy
    return stats
