import numpy as np
import sys

def get_conf_matrix(y_pred, y):
    nt_y = np.logical_not(y)
    nt_y_pred = np.logical_not(y_pred)
    tp = np.logical_and(y_pred, y).sum()
    tn = np.logical_and(nt_y_pred, nt_y).sum()
    fp = np.logical_and(y_pred, nt_y).sum()
    fn = np.logical_and(nt_y_pred, y).sum()
    conf_matrix = np.array([tp, fp, fn, tn]).reshape(2,2)
    return conf_matrix

def predict_random(y):
    import random
    m = y.shape[0]
    y_random = np.zeros((m,1))
    for i in range(m):
        y_random[i] = random.randint(0, 1)
    accuracy = (m-np.logical_xor(y_random, y).sum())/m
    return y_random

def predict_positive(y):
    m = y.shape[0]
    y_ones = np.ones((m,1))
    accuracy = (m-np.logical_xor(y_ones, y).sum())/m
    return y_ones

y = np.loadtxt('../y.csv', delimiter=',')
y_pred = np.loadtxt('../y_pred.csv', delimiter=',')
y = y.reshape(y.shape[0], 1)
y_pred = y_pred.reshape(y_pred.shape[0], 1)
y_random = predict_random(y)
y_positive = predict_positive(y)

print("Confusion matrix for Naive Bayes' Predictions:")
print(get_conf_matrix(y_pred, y))
print()
print("Confusion matrix for random Predictions:")
print(get_conf_matrix(y_random, y))
print()
print("Confusion matrix for positive Predictions:")
print(get_conf_matrix(y_positive, y))
print()