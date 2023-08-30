import numpy as np
import functions
import pickle
import sys

train_path = sys.argv[1]
infile = open(train_path + '/train_data.pickle','rb')
X_train, y_train = functions.preprocess(pickle.load(infile))

test_path = sys.argv[2]
infile = open(test_path + '/test_data.pickle','rb')
X_test, y_test = functions.preprocess(pickle.load(infile))
infile.close()

m_test = y_test.shape[0]
preds = functions.predict_all(X_train, y_train, X_test)
pred = np.zeros((m_test,1))
for i in range(m_test):
    pred[i][0] = np.bincount(preds[:,i]).argmax()

accuracy = (pred==y_test).sum()/m_test
np.savetxt('../pred_a.csv', pred, delimiter=',')
print("Test Accuracy is:")
print(accuracy*100)
print()