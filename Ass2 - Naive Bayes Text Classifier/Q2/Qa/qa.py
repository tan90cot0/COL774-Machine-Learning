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

alpha = functions.get_alpha(X_train,y_train)

w,b = functions.get_params(alpha, X_train, y_train)
print("Test Accuracy is")
functions.test(X_test, y_test, w,b)
functions.top5_alpha(alpha, X_train)
print()