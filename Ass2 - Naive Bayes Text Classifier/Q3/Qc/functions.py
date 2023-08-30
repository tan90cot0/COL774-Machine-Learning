import numpy as np
from PIL import Image
from sklearn import metrics
import matplotlib.pyplot as plt

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

def misclassified(y_test, X_test, pred):
    misclassified = []
    cnt = 0
    i = 0
    while cnt<10:
        if y_test[i]!=pred[i]:
            misclassified.append(X_test[i])
            cnt+=1
        i+=1
    w, h = 32, 32
    for l in range(10):
        test = misclassified[l]
        img_data = np.zeros((w, h, 3))
        for i in range(w):
            for j in range(h):
                for k in range(3):
                    img_data[i][j][k] = test[3*w*i + 3*j + k]*255

        img_data = img_data.astype('uint8')
        img = Image.fromarray(img_data, 'RGB')
        img.save('../plots/image{}.jpg'.format(l+1))

def conf_matrix(y_test, pred,l):
    pred = pred.flatten()
    y_test = y_test.flatten()
    conf_matrix = metrics.confusion_matrix(y_test, pred)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = conf_matrix, display_labels = [0,1,2,3,4])
    cm_display.plot()
    plt.savefig('../plots/conf_matrix{}.jpg'.format(l))