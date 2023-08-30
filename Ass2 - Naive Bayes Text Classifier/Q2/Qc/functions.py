import numpy as np

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

    return train_data, labels