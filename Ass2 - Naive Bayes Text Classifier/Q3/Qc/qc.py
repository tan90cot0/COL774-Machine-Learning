import numpy as np
import functions
import pickle
import sys

test_path = sys.argv[2]
infile = open(test_path + '/test_data.pickle','rb')
X_test, y_test = functions.preprocess(pickle.load(infile))
infile.close()

pred_a = np.loadtxt('../pred_a.csv',delimiter=',')
pred_b = np.loadtxt('../pred_b.csv',delimiter=',')

functions.conf_matrix(y_test, pred_a,1)
functions.misclassified(y_test, X_test, pred_a)
functions.conf_matrix(y_test, pred_b,2)
functions.misclassified(y_test, X_test, pred_b)