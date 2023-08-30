import numpy as np
from cvxopt import solvers
from cvxopt import matrix
from PIL import Image
import math

def K(x1,x2):
    return math.exp(-0.001*(np.linalg.norm(x1-x2))**2)

def get_alpha(X,y):
    m = y.shape[0]
    P = np.zeros((m,m))

    #outer product of y
    y = y.reshape(y.shape[0],1)
    outer = y@y.T
    for i in range(m):
        for j in range(m):
            P[i][j] = outer[i][j]*K(X[i], X[j])

    P = matrix(P, tc='d')
    q = matrix(np.ones((m,1))*-1, tc='d')
    G = matrix(np.vstack((np.eye(m), np.eye(m)*-1)), tc='d')
    H = matrix(np.vstack((np.ones((m,1)), np.zeros((m,1)))), tc='d')
    A = matrix(y.T, tc='d')
    b = matrix(np.array([0]), tc='d')

    sol = solvers.qp(P,q,G,H,A,b)
    alpha = np.array(sol['x'])
    return alpha

def sgn(x):
    if x>=0:
        return 1
    else:
        return -1

def get_b(alpha, X,y):
    m = X.shape[0]
    y2 = np.zeros((m,1))
    for i in range(m):
        y2[i] = predict2(X[i], 0, alpha, X, y)
    y3 = y[alpha>0]
    y4 = y2[alpha>0]
    b = np.mean(y3-y4)
    return b

def predict2(x,b,alpha,X,y):
    m = X.shape[0]
    s = b
    a = alpha*y
    for i in range(m):
        s+=a[i]*K(X[i], x)
    return s

def test(X,alpha,b, X_train, y_train):
    m = X.shape[0]
    y_predict = np.zeros(m)
    for i in range(m):
        y_predict[i] = sgn(predict2(X[i], b, alpha, X_train, y_train))
    y_predict[y_predict>=0] = 1
    y_predict[y_predict<0] = -1
    return y_predict

def display_top5_alpha(alpha, X):
    t = np.sort(alpha)[:5]
    top_5 = np.logical_or(alpha==t[0], np.logical_or(alpha==t[1], np.logical_or(alpha==t[2], np.logical_or(alpha==t[3], alpha==t[4]))))
    top5 = []
    for i in range(top_5.shape[0]):
        if top_5[i]==1:
            top5.append(i)
    top5 = np.array(top5)
    best_x = X[top5]

    for i in range(best_x.shape[0]):
        w, h = 32, 32
        test = best_x[i]
        img_data = np.zeros((w, h, 3))
        for i in range(w):
            for j in range(h):
                for k in range(3):
                    img_data[i][j][k] = test[3*w*i + 3*j + k]*255

        img_data = img_data.astype('uint8')
        img = Image.fromarray(img_data, 'RGB')
        img.show()
        #display(img.resize((w,h)))


def predict(X_train,y_train,X_test, label1, label2):
    y_pred = predict_cvxopt(X_train, y_train, X_test)
    y_pred[y_pred==1] = label2
    y_pred[y_pred==-1] = label1
    return y_pred

def predict_all(X_train, y_train, X_test):
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
            preds[index] = predict(X,y, X_test,i,j)
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

    return train_data, labels.reshape(labels.shape[0],1)

def get_data(label1, label2, X, y):
    new_examples = (X.shape[0]*2)//5
    train_data = np.zeros((new_examples, X.shape[1]))
    index = 0
    indices = np.logical_or(y==label1, y==label2)
    for i in range(X.shape[0]):
        if indices[i]:
            train_data[index,:] = X[i]
            index+=1

    y = y[indices]
    y = y.astype('float64')
    y[y==label1] = -1
    y[y==label2] = 1
    return train_data,y.reshape(y.shape[0],1)


def predict_cvxopt(X_train, y_train, X_test):
    alpha = get_alpha(X_train,y_train)
    alpha = alpha.reshape(X_train.shape[0], 1)
    b = get_b(alpha, X_train, y_train)
    return test(X_test, alpha,b, X_train, y_train)