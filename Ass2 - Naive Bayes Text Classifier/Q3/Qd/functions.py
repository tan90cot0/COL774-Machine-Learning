import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC


def plot_logplot(test, valid, logc):
    plt.plot(logc, test, color = 'blue', label = "Test", marker = "o")
    plt.plot(logc, valid, color = 'red', label = "Cross Validation", marker = "x")
    plt.xlabel("log C")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("../plots/3d.png")

def predict(X_train,y_train,X_test, label1, label2, c):
    svm = SVC(kernel='rbf', random_state=1, C=c, gamma = 0.001)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    y_pred[y_pred==1] = label2
    y_pred[y_pred==-1] = label1
    return y_pred

def predict_all(X_train, y_train, X_test,c):
    train_data = []
    for i in range(5):
        for j in range(i+1,5):
            train_data.append(get_data(i,j,X_train, y_train))
        
    index = 0
    m_test = X_test.shape[0]
    preds = np.zeros((10,m_test))
    for i in range(5):
        for j in range(i+1,5):
            X,y = train_data[index]
            preds[index] = predict(X,y, X_test,i,j,c)
            index+=1
    preds = preds.astype('int64')
    return preds

def preprocess(data_dict):
    data = data_dict["data"]
    labels = data_dict["labels"]
    dims = data.shape
    train_data = np.zeros((dims[0], dims[1]*dims[2]*dims[3]))
    
    for l in range(dims[0]):
        x = []
        for i in range(dims[1]):
            for j in range(dims[2]):
                for k in range(dims[3]):
                    x.append(data[l][i][j][k]/255)
        train_data[l,:] = np.array(x)

    return train_data, labels

def get_data(label1, label2, X, y):
    indices = np.logical_or(y==label1, y==label2)
    new_examples = indices.sum()
    train_data = np.zeros((new_examples, X.shape[1]))
    index = 0
    for i in range(X.shape[0]):
        if indices[i]:
            train_data[index,:] = X[i]
            index+=1

    y = y[indices]
    y = y.astype('float64')
    y[y==label1] = -1
    y[y==label2] = 1
    return train_data,y

def model(data,c):
    X_train, y_train, X_test, y_test = data
    m_test = y_test.shape[0]
    preds = predict_all(X_train, y_train, X_test,c)
    pred = np.zeros((m_test,1))
    for i in range(m_test):
        pred[i][0] = np.bincount(preds[:,i]).argmax()
    accuracy = (pred==y_test).sum()/m_test
    return accuracy*100

def get_validation_data(full_data,i):
    m = full_data.shape[0]
    batch = m//5
    cross_valid_data = full_data[i*batch:(i+1)*batch]
    train_data = np.vstack((full_data[:i*batch], full_data[(i+1)*batch:]))
    X_new_train = train_data[:, :-1]
    y_new_train = train_data[:, -1].reshape(X_new_train.shape[0], 1)
    X_new_test = cross_valid_data[:, :-1]
    y_new_test = cross_valid_data[:, -1].reshape(X_new_test.shape[0], 1)
    return X_new_train, y_new_train, X_new_test, y_new_test