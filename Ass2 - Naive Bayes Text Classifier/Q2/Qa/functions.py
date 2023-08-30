import numpy as np
from cvxopt import solvers
from cvxopt import matrix
from PIL import Image

def get_alpha(X,y):
    m = y.shape[0]
    Xy = X*y

    P = matrix(Xy@Xy.T, tc='d')
    q = matrix(np.ones((m,1))*-1, tc='d')
    G = matrix(np.vstack((np.eye(m), np.eye(m)*-1)), tc='d')
    H = matrix(np.vstack((np.ones((m,1)), np.zeros((m,1)))), tc='d')
    A = matrix(y.T, tc='d')
    b = matrix(np.array([0]), tc='d')

    sol = solvers.qp(P,q,G,H,A,b)
    alpha = np.array(sol['x'])
    return alpha

def get_params(alpha, X,y):
    alpha[np.logical_and(alpha<=1e-5, alpha>=1-1e-5)] = 0
    alpha = alpha.reshape(alpha.shape[0], 1)
    temp = (alpha*y*X)
    w = temp.sum(axis = 0).reshape(X.shape[1], 1)
    Xw = X@w
    b = np.mean(y-Xw)
    return w,b

def test(X,y,w,b):
    m = X.shape[0]
    y_predict = X@w + b
    y_predict[y_predict>=0] = 1
    y_predict[y_predict<0] = -1
    wrong = (np.abs(y_predict-y)/2).sum()
    accuracy = 1-wrong/m
    print(accuracy*100)

def top5_alpha(alpha, X):
    t = np.sort(alpha)[:5]
    top_5 = np.logical_or(alpha==t[0], np.logical_or(alpha==t[1], np.logical_or(alpha==t[2], np.logical_or(alpha==t[3], alpha==t[4]))))
    top5 = []
    for i in range(top_5.shape[0]):
        if top_5[i]==1:
            top5.append(i)
    top5 = np.array(top5)
    best_x = X[top5]

    for l in range(best_x.shape[0]):
        w, h = 32, 32
        test = best_x[l]
        img_data = np.zeros((w, h, 3))
        for i in range(w):
            for j in range(h):
                for k in range(3):
                    img_data[i][j][k] = test[3*w*i + 3*j + k]*255

        img_data = img_data.astype('uint8')
        img = Image.fromarray(img_data, 'RGB')
        img.save('../plots/linear_best5_{}.jpg'.format(l+1))

def preprocess(data_dict):
    data = data_dict["data"]
    labels = data_dict["labels"]
    dims = data.shape
    new_dim = dims[1]*dims[2]*dims[3]
    new_examples = (data.shape[0]*2)//5
    train_data = np.zeros((new_examples, new_dim))
    
    
    index = 0
    for l in range(dims[0]):
        if labels[l][0]==0 or labels[l][0]==1:
            x = []
            for i in range(dims[1]):
                for j in range(dims[2]):
                    for k in range(dims[3]):
                        x.append(data[l][i][j][k]/255)
            train_data[index,:] = np.array(x)
            index+=1
    
    labels = labels[np.logical_or(labels==0, labels==1)]
    labels = labels.astype('float64')
    labels[labels==0] = -1

    return train_data, labels.reshape(labels.shape[0],1)