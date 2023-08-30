import numpy as np

def predict_random(y):
    import random
    m = y.shape[0]
    y_random = np.zeros((m,1))
    for i in range(m):
        y_random[i] = random.randint(0, 1)
    accuracy = (m-np.logical_xor(y_random, y).sum())/m
    print("Random Prediction Accuracy is: " + str(np.round(accuracy*100, 2)) + " percent")
    return y_random

def predict_positive(y):
    m = y.shape[0]
    y_ones = np.ones((m,1))
    accuracy = (m-np.logical_xor(y_ones, y).sum())/m
    print("Positive Prediction Accuracy is: " + str(np.round(accuracy*100, 2)) + " percent")
    return y_ones

y = np.loadtxt('../y.csv', delimiter=',')
y = y.reshape(y.shape[0], 1)
predict_random(y)
print()
predict_positive(y)
print()