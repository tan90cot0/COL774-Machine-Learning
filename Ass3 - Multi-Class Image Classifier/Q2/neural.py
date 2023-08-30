import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import time
from sklearn.neural_network import MLPClassifier
import sys

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_der(x):
    return sigmoid(x)*sigmoid(1-x)

def relu_func(x):
    return x*(x>0)

#check this
def relu_der(x):
    return 1*(x>0)

def cost_function(a,y):
    m = y.shape[1]
    matrix = (y-a)*(y-a)
    return (1/(2*m))*(np.sum(matrix))

def get_data(path):
    df = pd.read_csv (path, header=None)
    pixel_values = df.iloc[0:df.shape[0],0:df.shape[1]-1]
    pixel_values = np.array(pixel_values)

    #Data normalization
    max_pixel_value = np.max(pixel_values)
    pixel_values = pixel_values/max_pixel_value

    temp = df.iloc[:,df.shape[1]-1]
    temp = np.array(temp)

    output = np.zeros((df.shape[0],10))
    for i in range(10):
        output[np.where(temp==i),i]=1
    return pixel_values.T, output.T

def init_params(layer):
    W = []
    b = []
    for i in range(len(layer)-1):
        np.random.seed(i)
        W.append(np.random.randn(layer[i+1],layer[i])*0.01)
        b.append(np.zeros((layer[i+1],1)))
    return W,b

def minibatch_create(X, y, r):
    X = X.T
    y = y.T
    minibatches = []
    joint = np.hstack((X, y))
    #to randomise the combined dataset
    np.random.shuffle(joint)
    for i in range(joint.shape[0]//r):
        minibatch = joint[i * r:(i + 1)*r, :]
        new_X = minibatch[:, :-10].T
        new_y = minibatch[:, -10:].reshape((-1, 10)).T
        minibatches.append((new_X, new_y))
    return minibatches

def forward_prop(params,X, relu):
    W,b = params
    a = [X]
    z = []
    for i in range(len(b)):
        z.append(W[i]@a[i] + b[i])
        if relu==True and i<len(b)-1:
            a.append(relu_func(z[i]))
        else:
            a.append(sigmoid(z[i]))
    return z,a

def derivative(z, relu):
    return relu_der(z) if relu==True else sigmoid_der(z)

def outer_loss(mse, y, al,z):
    if mse:
        return (al-y)*sigmoid_der(z)
    else:
        return ((1-y)*sigmoid(z) - y*sigmoid(1-z))

def back_prop(params, z, a, y, relu, mse):
    W,b = params
    delta = []
    for i in range(len(b)):
        delta.append(None)
    grads = {'W':[], 'b':[]}
    m = y.shape[1]
    
    for i in range(len(b)-1,-1,-1): 
        if i==len(b)-1:
            delta[i] = outer_loss(mse, y, a[i+1],z[i])
        else:
            delta[i] = np.dot(W[i+1].T, delta[i+1]) * (relu_der(z[i]) if relu==True else sigmoid_der(z[i]))
        grads['W'].append(np.dot(delta[i], a[i].T)/m)
        grads['b'].append(np.sum(delta[i], axis=1, keepdims=True)/m)
    grads['W'].reverse()
    grads['b'].reverse()
    
    return grads

def update(params,grads,alpha):
    W,b = params
    for i in range(len(W)):
        W[i] = W[i] - alpha*grads['W'][i]
        b[i] = b[i] - alpha*grads['b'][i]
    return W,b

def accuracy(y,y_hat):
    matrix = y - y_hat
    positive = 0
    for i in range(y.shape[1]):
        if(np.max(matrix[:,i])==0):
            positive = positive + 1
    return (positive/y.shape[1])*100

def model(max_epochs, minibatches, params, alpha, adaptive, conv, relu, mse):
    cost_history = []
    for i in range(max_epochs):
        if adaptive:
            lr = alpha/np.sqrt(i+1)
        else:
            lr = alpha
            
        for X,y in minibatches:
            z,a = forward_prop(params, X , relu)
            grads = back_prop(params, z, a, y , relu, mse)
            params = update(params, grads, lr)

        cost_history.append(cost_function(a[-1], y))
        if i>1 and (abs(cost_history[-1] - cost_history[-2]) < conv):
            return params
    return params

def plot_confusion_matrix(y, y_pred, num, part, output_path):
    cm_display = ConfusionMatrixDisplay.from_predictions(y_pred.T.argmax(axis = 1), y.T.argmax(axis = 1))
    if part=='d':
        if num==0:
            plt.savefig(output_path + '/CM_Part_{}_Relu.jpg'.format(part))
            plt.clf()
        else:
            plt.savefig(output_path + '/CM_Part_{}_Sigmoid.jpg'.format(part))
            plt.clf()
                        
    else:
        plt.savefig(output_path + '/CM_Part_{}_HiddenSize_{}.jpg'.format(part, num))
        plt.clf()
    
def predict(params, X, relu):
    z,a = forward_prop(params,X, relu)
    y_pred = a[-1]
    m = y_pred.shape[1]
    for i in range(m):
        maximum = np.max(y_pred[:,i])
        for j in range(10):
            if(y_pred[j,i]==maximum):
                y_pred[j,i]=1
            else:
                y_pred[j,i]=0
    return y_pred

def plot_graphs(train_accuracies, test_accuracies, training_times, part, output_path):
    layers = [5,10,15,20,25]
    plt.title("Accuracies vs Layer Size")
    plt.xlabel("Hidden layer size(units)")
    plt.ylabel("Accuracy")
    plt.plot(layers, train_accuracies)
    plt.plot(layers, test_accuracies)
    plt.legend(["Training Accuracies", "Test Accuracies"])
    plt.savefig(output_path + "/Part_{}_Accuracies.png".format(part))
    plt.clf()
                
    plt.title("Training Time vs Layer Size")
    plt.xlabel("Hidden layer size(units)")
    plt.ylabel("Training Time")
    plt.plot(layers, training_times)
    plt.legend(["Training Time"])
    plt.savefig(output_path + "/Part_{}_TrainingTime.png".format(part))
    plt.clf()
    
def train_and_test(batch_size, layer_dims, max_epochs, alpha, paths, i, adaptive, conv, part, mse, output_path):
    if (part=='d' or part=='e' or part=='f') and i==0:
        relu = True
    else:
        relu = False
    X_train, y_train= get_data(paths[0])
    minibatches = minibatch_create(X_train, y_train, batch_size)
    params = init_params(layer_dims)
    begin = time.time()
    params = model(max_epochs, minibatches, params, alpha, adaptive, conv, relu, mse)
    time_to_train = time.time() - begin
    y_pred = predict(params, X_train, relu)
    acc_train = accuracy(y_train,y_pred)
    
    X_test, y_test= get_data(paths[1])
    y_pred_test = predict(params, X_test, relu)
    acc_test = accuracy(y_test,y_pred_test)
    
    if part=='d':
        plot_confusion_matrix(y_test, y_pred_test, i, part, output_path)
    elif part=='b' or part=='c':
        plot_confusion_matrix(y_test, y_pred_test, 5*(i+1), part, output_path)
    
    return acc_train, acc_test, time_to_train

def part_b(paths, output_path):
    print("b")
    batch_size = 100
    alpha = 0.1
    max_epochs = 500

    train_accuracies = []
    test_accuracies = []
    training_times = []
    hidden_layer_vals = [[5], [10], [15], [20], [25]]
    adaptive = False 
    conv_criteria = 1e-9
    lines = []

    for i in range(len(hidden_layer_vals)):
        layer_dims = [784] + hidden_layer_vals[i] + [10]
        acc_train, acc_test, time_to_train = train_and_test(batch_size, layer_dims, max_epochs, alpha ,paths, i, adaptive, conv_criteria, 'b', True, output_path)
        train_accuracies.append(acc_train)
        test_accuracies.append(acc_test)
        training_times.append(time_to_train)
        lines.append("For hidden layer size {}: \n".format(hidden_layer_vals[i][0]))
        lines.append("Training Accuracy = " + str(np.round(acc_train,2)) + " percent\n")
        lines.append("Test Accuracy = " + str(np.round(acc_test,2)) + " percent\n")
        lines.append("Time taken to train = " + str(np.round(time_to_train,2)) + " seconds\n")
        lines.append("\n")
        
    plot_graphs(train_accuracies, test_accuracies, training_times, 'b', output_path)
    
    return lines
    
def part_c(paths, output_path):
    print("c")
    batch_size = 100
    alpha = 0.1
    max_epochs = 500

    train_accuracies = []
    test_accuracies = []
    training_times = []
    hidden_layer_vals = [[5], [10], [15], [20], [25]]
    adaptive = True
    conv_criteria = 1e-9

    lines = []

    for i in range(len(hidden_layer_vals)):
        layer_dims = [784] + hidden_layer_vals[i] + [10]
        acc_train, acc_test, time_to_train = train_and_test(batch_size, layer_dims, max_epochs, alpha ,paths, i, adaptive, conv_criteria, 'c', True, output_path)
        train_accuracies.append(acc_train)
        test_accuracies.append(acc_test)
        training_times.append(time_to_train)
        lines.append("For hidden layer size {}: \n".format(hidden_layer_vals[i][0]))
        lines.append("Training Accuracy = " + str(np.round(acc_train,2)) + " percent\n")
        lines.append("Test Accuracy = " + str(np.round(acc_test,2)) + " percent\n")
        lines.append("Time taken to train = " + str(np.round(time_to_train,2)) + " seconds\n")
        lines.append("\n")
        
    plot_graphs(train_accuracies, test_accuracies, training_times, 'c', output_path)
    
    return lines
                    
def part_d(paths, output_path):
    print("d")
    batch_size = 100
    alpha = 0.1
    max_epochs = 500

    hidden_layer = [100,100]
    adaptive = True
    conv_criteria = 1e-9

    layer_dims = [784] + hidden_layer + [10]
    acc_train, acc_test, time_to_train = train_and_test(batch_size, layer_dims, max_epochs, alpha ,paths, 0, adaptive, conv_criteria, 'd', True, output_path)

    
    lines = []
    lines.append("For Activation {}: \n".format("Relu"))
    lines.append("Training Accuracy = " + str(np.round(acc_train,2)) + " percent\n")
    lines.append("Test Accuracy = " + str(np.round(acc_test,2)) + " percent\n")
    lines.append("\n")
                    
    acc_train, acc_test, time_to_train = train_and_test(batch_size, layer_dims, max_epochs, alpha ,paths, 1, adaptive, conv_criteria, 'd', True, output_path)

    lines.append("For Activation {}: \n".format("Sigmoid"))
    lines.append("Training Accuracy = " + str(np.round(acc_train,2)) + " percent\n")
    lines.append("Test Accuracy = " + str(np.round(acc_test,2)) + " percent\n")
    
    return lines
    
def part_e(paths, output_path):
    print("e")
    batch_size = 100
    alpha = 0.1
    max_epochs = 500

    hidden_layer_vals = [[50,50], [50,50,50], [50,50,50,50], [50,50,50,50,50]]
    adaptive = False
    conv_criteria = 1e-9
    
    train_accuracies = []
    test_accuracies = []

    lines = []
    lines.append("For Relu\n")
    for i in range(len(hidden_layer_vals)):
        layer_dims = [784] + hidden_layer_vals[i] + [10]
        acc_train, acc_test, time_to_train = train_and_test(batch_size, layer_dims, max_epochs, alpha ,paths, 0, adaptive, conv_criteria, 'e', True, output_path)
        train_accuracies.append(acc_train)
        test_accuracies.append(acc_test)
        lines.append("For Number of Layers {}: \n".format(len(hidden_layer_vals[i])))
        lines.append("Training Accuracy = " + str(np.round(acc_train,2)) + " percent\n")
        lines.append("Test Accuracy = " + str(np.round(acc_test,2)) + " percent\n")
        lines.append("\n")
        
    layers = [2,3,4,5]
    plt.title("Accuracies vs Number of Layers")
    plt.xlabel("Number of Layers")
    plt.ylabel("Accuracy")
    plt.plot(layers, train_accuracies)
    plt.plot(layers, test_accuracies)
    plt.legend(["Training Accuracies", "Test Accuracies"])
    plt.savefig(output_path + "/Part_e_Accuracies_Relu.png")
    plt.clf()
    
    train_accuracies = []
    test_accuracies = []

    lines.append("For Sigmoid\n")
    for i in range(len(hidden_layer_vals)):
        layer_dims = [784] + hidden_layer_vals[i] + [10]
        acc_train, acc_test, time_to_train = train_and_test(batch_size, layer_dims, max_epochs, alpha ,paths, 1, adaptive, conv_criteria, 'e', True, output_path)
        train_accuracies.append(acc_train)
        test_accuracies.append(acc_test)
        lines.append("For Number of Layers {}: \n".format(len(hidden_layer_vals[i])))
        lines.append("Training Accuracy = " + str(np.round(acc_train,2)) + " percent\n")
        lines.append("Test Accuracy = " + str(np.round(acc_test,2)) + " percent\n")
        lines.append("\n")
        
    layers = [2,3,4,5]
    plt.title("Accuracies vs Number of Layers")
    plt.xlabel("Number of Layers")
    plt.ylabel("Accuracy")
    plt.plot(layers, train_accuracies)
    plt.plot(layers, test_accuracies)
    plt.legend(["Training Accuracies", "Test Accuracies"])
    plt.savefig(output_path + "/Part_e_Accuracies_Sigmoid.png")
    plt.clf()
    
    return lines
    
def part_f(paths, output_path):
    print('f')
    batch_size = 100
    alpha = 0.1
    max_epochs = 500

    #change this
    hidden_layer = [50,50]
    adaptive = True
    conv_criteria = 1e-9

    layer_dims = [784] + hidden_layer + [10]
    acc_train, acc_test, time_to_train = train_and_test(batch_size, layer_dims, max_epochs, alpha ,paths, 0, adaptive, conv_criteria, 'f', False, output_path)

    lines = []

    lines.append("For BCE: \n")
    lines.append("Training Accuracy = " + str(np.round(acc_train,2)) + " percent\n")
    lines.append("Test Accuracy = " + str(np.round(acc_test,2)) + " percent\n")
    lines.append("\n")
    
    return lines

def part_g(paths, output_path):
    print('g')
    X_train, y_train= get_data(paths[0])
    X_test, y_test= get_data(paths[1])

    clf = MLPClassifier(hidden_layer_sizes = (50,50), 
                        solver= 'sgd', 
                        batch_size = 100,
                        learning_rate = 'adaptive',
                        learning_rate_init = 0.1,
                        random_state=1, 
                        max_iter=500).fit(X_train.T, y_train.argmax(axis=0))
    
    lines = []
    lines.append("Training Accuracy = " + str(np.round(clf.score(X_train.T, y_train.argmax(axis=0))*100,2)) + " percent\n")
    lines.append("Test Accuracy = " + str(np.round(clf.score(X_test.T, y_test.argmax(axis=0))*100,2)) + " percent\n")
    
    return lines


def init():
    train_path = sys.argv[1]
    test_path = sys.argv[2]
    paths = [train_path,test_path]
    output_path = sys.argv[3]
    part = sys.argv[4]

    part_dict = {'b':part_b, 'c':part_c, 'd': part_d, 'e': part_e, 'f':part_f, 'g': part_g}

    result = part_dict[part](paths, output_path)

    file = open(output_path + '/' + part + ".txt", "w")
    file.writelines(result)
    file.close()

init()