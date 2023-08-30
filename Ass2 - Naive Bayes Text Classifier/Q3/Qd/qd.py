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

data_original = X_train, y_train, X_test, y_test

full_data = np.hstack((X_train, y_train))
np.random.shuffle(full_data)

C = [1e-5, 1e-3, 1, 5, 10]
cross_accuracies = []
test_accuracies = []
for j in range(len(C)):
    cross_accuracies.append([])
    for i in range(5):
        data = functions.get_validation_data(full_data, i)
        acc = functions.model(data,C[j])
        cross_accuracies[j].append(acc)
    acc = functions.model(data_original,C[j])
    test_accuracies.append(acc)

validation_acc = []
for i in range(5):
    s = 0
    for j in range(5):
        s+=cross_accuracies[i][j]
    validation_acc.append(s/5)

C = [1e-5, 1e-3, 1, 5, 10]
for i in range(len(C)):
    print("Test Accuracy for C = " + str(C[i]))
    print(test_accuracies[i])
    print("5-fold Cross-Validation Accuracy for C = " + str(C[i]))
    print(validation_acc[i])
    print()
logc = [-5, -3, 0, 0.7, 1]
functions.plot_logplot(test_accuracies, validation_acc, logc)