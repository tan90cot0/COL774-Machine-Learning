from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
import pickle
import functions
import sys


train_path = sys.argv[1]
infile = open(train_path + '/train_data.pickle','rb')
X_train, y_train = functions.preprocess(pickle.load(infile))

test_path = sys.argv[2]
infile = open(test_path + '/test_data.pickle','rb')
X_test, y_test = functions.preprocess(pickle.load(infile))
infile.close()

svm = SVC(kernel = "linear", random_state=1, C=1, gamma=0.001)
svm.fit(X_train, y_train)
 
y_pred = svm.predict(X_test)
print("Test Accuracy is")
print('%.3f' % accuracy_score(y_test, y_pred))
print()