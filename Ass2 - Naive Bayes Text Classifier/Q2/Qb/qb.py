import pickle
import functions
import numpy as np
import sys

train_path = sys.argv[1]
infile = open(train_path + '/train_data.pickle','rb')
X_train, y_train = functions.preprocess(pickle.load(infile))

test_path = sys.argv[2]
infile = open(test_path + '/test_data.pickle','rb')
X_test, y_test = functions.preprocess(pickle.load(infile))
infile.close()

functions.predict_cvxopt(X_train, y_train, X_test, y_test)
print()