import pandas as pd
import math
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import coo_matrix, hstack, csr_matrix
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import time
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import HalvingRandomSearchCV
import sys

def clean(review, stopwords):
    tokens = review.lower().replace('&#039;', "").replace('.', " ").replace(',', " ").replace('"', " ").replace('\r', " ").replace('\n', " ").replace('-', " ").split()
    new_review = ""
    for word in tokens:
        if word not in stopwords:
            new_review+=word + " "
    return new_review[:-1]

def get_vocab(st):
    vectorizer = CountVectorizer()
    st = vectorizer.fit_transform(st)
    vocab = vectorizer.get_feature_names_out()
    return st, vocab, vectorizer

def extract_string_feature(feature, stop_words, train, vectorizer):
    for i in range(len(feature)):
        if type(feature[i])==float and math.isnan(feature[i]):
            feature[i] = ""
        else:
            feature[i] = clean(feature[i], stop_words)

    if train:
        return get_vocab(feature)
    else:
        return vectorizer.transform(feature), None, vectorizer

def get_date(l):
    months = {"January": 1, "February":2, "March":3, "April":4, "May":5, "June":6, "July":7, "August":8, "September":9, "October":10, "November":11, "December":12}
    dates = np.zeros((len(l), 3))
    for i in range(len(l)):
        temp = l[i].split()
        month = months[temp[0]]
        date = int(temp[1][:-1])
        year = int(temp[2])
        dates[i] = np.array([date, month, year])
    return csr_matrix(dates)

def get_arrays(data, train, vect, stop_words):
    conditions, cond_vocab, vect[0] = extract_string_feature(list(data["condition"]), stop_words, train, vect[0])
    reviews, reviews_vocab, vect[1] = extract_string_feature(list(data["review"]), stop_words, train, vect[1])
    dates = get_date(list(data["date"])) 
    usefulCount = csr_matrix(np.array(data['usefulCount']).reshape(dates.shape[0], 1))
    X = hstack([reviews, conditions, dates, usefulCount])
    y = np.array(data["rating"])
    y = y-1
    #THIS CHANGED

    return (X,y),vect

def get_data(paths):
    stop_words = set(stopwords.words('english'))
    vect = [None, None]
    full_data = []
    for i in range(3):
        data, vect = get_arrays(pd.read_csv(paths[i]), i==0, vect, stop_words)
        full_data.append(data)

    return tuple(full_data)

def plot_alpha_graphs(alphas, metrics, scores, output_path):
    train_scores, validation_scores, test_scores = scores
    plt.xlabel("Alphas")
    plt.ylabel("Accuracies")
    plt.title("Accuracy vs Alpha for Training and Test sets")
    plt.plot(alphas, train_scores, marker="o", label="Train", drawstyle="steps-post")
    plt.plot(alphas, validation_scores, marker="o", label="Validation", drawstyle="steps-post")
    plt.plot(alphas, test_scores, marker="o", label="Test", drawstyle="steps-post")
    plt.legend()
    plt.savefig(output_path + '/Accuracy_2')
    plt.clf()

    impurities, nodes, depth = metrics
    
    plt.xlabel("Alphas")
    plt.ylabel("Impurities")
    plt.title("Impurity vs Alpha")
    plt.plot(alphas, impurities, marker="o", label="Impurities", drawstyle="steps-post")
    plt.legend()
    plt.savefig(output_path + '/Impurity_2')
    plt.clf()


    plt.xlabel("Alphas")
    plt.ylabel("Nodes")
    plt.title("Nodes vs Alpha")
    plt.plot(alphas, nodes, marker="o", label="Nodes", drawstyle="steps-post")
    plt.legend()
    plt.savefig(output_path + '/Nodes_2')
    plt.clf()

    plt.xlabel("Alphas")
    plt.ylabel("Depth")
    plt.title("Depth vs Alpha")
    plt.plot(alphas, depth, marker="o", label="Depth", drawstyle="steps-post")
    plt.legend()
    plt.savefig(output_path + '/Depth_2')
    plt.clf()

def model(clf, data):
    X_train,y_train = data[0]
    clf.fit(X_train, y_train)
    acc = []
    for X,y in data:
        acc.append(clf.score(X,y))
    return acc

def part_a(data, output_path):
    X_train, y_train = data[0]
    clf = tree.DecisionTreeClassifier(random_state = 0)
    acc = model(clf, data)
    lines = ['Training Accuracy is: ', 'Validation Accuracy is: ', 'Test Accuracy is: ']
    lines = [lines[i] + str(np.round(acc[i]*100,5)) + '\n' for i in range(len(lines))]
    lines = ["Results for Part a:\n"] + lines + ["\n"]
    return lines, acc

def get_part_b_params(train_data):
    clf = tree.DecisionTreeClassifier()
    param_dict = {"max_depth": range(10,50,10), 
                 "min_samples_split": range(20,100,10), 
                 "min_samples_leaf": range(10,50,10)
                 }

    grid = GridSearchCV(clf, param_grid = param_dict, cv = 10, verbose = 1, n_jobs = -1)
    X_train,y_train = train_data
    grid.fit(X_train, y_train)
    p = grid.best_params_
    return p['max_depth'], p['min_samples_leaf'], p['min_samples_split']

def part_b(data, output_path):
    params = get_part_b_params(data[0])
    X_train, y_train = data[0]
    clf = tree.DecisionTreeClassifier(random_state = 0, max_depth = params[0], min_samples_leaf = params[1], min_samples_split = params[2])
    acc = model(clf, data)
    lines = ['Training Accuracy is: ', 'Validation Accuracy is: ', 'Test Accuracy is: ']
    lines = [lines[i] + str(np.round(acc[i]*100,5)) + '\n' for i in range(len(lines))]
    lines = ["Results for Part b:\n"] + lines + ["\n"]
    return lines, acc

def get_scores(data, clfs):
    train_data, validation_data, test_data = data
    X_train, y_train = train_data
    X_validation, y_validation = validation_data
    X_test, y_test = test_data 
    train_scores = [clf.score(X_train, y_train) for clf in clfs]
    validation_scores = [clf.score(X_validation, y_validation) for clf in clfs]
    test_scores = [clf.score(X_test, y_test) for clf in clfs]
    
    return train_scores, validation_scores, test_scores

def get_part_c_params(train_data):
    X_train, y_train = train_data
    clf = tree.DecisionTreeClassifier(random_state=0)
    path = clf.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities

    clfs = []
    nodes = []
    depth = []
    for ccp_alpha in ccp_alphas:
        clf = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
        clf.fit(X_train, y_train)
        clfs.append(clf)
        nodes.append(clf.tree_.node_count)
        depth.append(clf.tree_.max_depth)

    ccp_alphas = ccp_alphas[:-1]
    clfs = clfs[:-1]
    impurities = impurities[:-1]
    nodes = nodes[:-1]
    depth = depth[:-1]

    metrics = (impurities, nodes, depth)
    
    return ccp_alphas, clfs, metrics

def part_c_new(data, output_path):
    train_data, validation_data, test_data = data
    X_train, y_train = train_data
    X_validation, y_validation = validation_data
    X_test, y_test = test_data 

    clf = tree.DecisionTreeClassifier(random_state=0)
    path = clf.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities

    l = []
    x_u = len(ccp_alphas)-1
    x_l = 0
    gr = (math.sqrt(5) + 1)/2
    while (x_u - x_l)>5:
        c = int(x_u - (x_u - x_l)//gr)
        d = int(x_l + (x_u - x_l)//gr)
        
        clf1 = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alphas[c]).fit(X_train, y_train)
        fc = clf1.score(X_validation, y_validation)*100
        temp1 = clf1.score(X_train, y_train)*100
        temp2 = clf1.score(X_test, y_test)*100
        l.append([ccp_alphas[c], temp1, fc ,temp2, impurities[c], clf1.tree_.node_count, clf1.tree_.max_depth])
        
        clf2 = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alphas[d]).fit(X_train, y_train)
        fd = clf2.score(X_validation, y_validation)*100
        temp1 = clf2.score(X_train, y_train)*100
        temp2 = clf2.score(X_test, y_test)*100
        l.append([ccp_alphas[d], temp1, fd ,temp2, impurities[d], clf2.tree_.node_count, clf2.tree_.max_depth])
        
        if fc>fd:
            x_u=d
        else:
            x_l=c
    ind = (x_u+x_l)//2
    best_alpha = ccp_alphas[ind]
    clf = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=best_alpha).fit(X_train, y_train)

    temp1 = clf.score(X_train, y_train)*100
    temp2 = clf.score(X_validation, y_validation)*100
    temp3 = clf.score(X_test, y_test)*100

    l.append([best_alpha, temp1, temp2, temp3, impurities[ind], clf.tree_.node_count, clf.tree_.max_depth])
    l.sort()

    training = []
    test = []
    valid = []
    alphas = []
    imp = []
    nodes = []
    depth = []
    for i in range(len(l)):
        alphas.append(l[i][0])
        training.append(l[i][1])
        valid.append(l[i][2])
        test.append(l[i][3])
        imp.append(l[i][4])
        nodes.append(l[i][5])
        depth.append(l[i][6])
        
    metrics = (imp, nodes, depth)
    scores = (training, valid, test)
    plot_alpha_graphs(alphas, metrics, scores, output_path)

    lines = ['Training Accuracy is: ', 'Validation Accuracy is: ', 'Test Accuracy is: ']
    lines = [lines[i] + str(np.round(clf.score(data[i][0], data[i][1])*100,5)) + '\n' for i in range(3)]

    lines.append('Best Alpha = ' + str(np.round(best_alpha,5)) + '\n')
    lines = ["Results for Part c:\n"] + lines + ["\n"]
    
    return lines, [temp1, temp2, temp3]

def part_c(data, output_path):
    ccp_alphas, clfs, metrics = get_part_c_params(data[0])
    scores = get_scores(data, clfs)

    plot_alpha_graphs(ccp_alphas, metrics, scores)
    index = np.argmax(np.array(scores[1]))
    best_alpha, best_tree = ccp_alphas[index], clfs[index]
    acc = scores[0][index], scores[1][index], scores[2][index]
    
    lines = ['Training Accuracy is: ', 'Validation Accuracy is: ', 'Test Accuracy is: ']
    lines = [lines[i] + str(np.round(acc[i]*100,5)) + '\n' for i in range(len(lines))]
    lines.append('Best Alpha = ' + str(np.round(best_alpha,5)) + '\n')
    lines = ["Results for Part c:\n"] + lines + ["\n"]
    return lines, acc

def get_part_d_params(X,y):
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    #check max features
    l = [0.4, 0.5, 0.6, 0.7, 0.8]
    param_dict = {"n_estimators": range(50,450, 50), 
                    "max_features": ["sqrt", "log2", None], 
                    "min_samples_split": range(2,10,2)
                    }
    grid = GridSearchCV(clf, param_grid = param_dict, cv = 10, verbose = 1, n_jobs = -1)
    grid.fit(X,y)
    p = grid.best_params_
    return p["n_estimators"], p['min_samples_split'], p['max_features']

def part_d(data, output_path):
    X_train,y_train = data[0]
    params = get_part_d_params(X_train, y_train)
    clf = RandomForestClassifier(max_depth=2, random_state=0, n_estimators = params[0], min_samples_split = params[1], max_features = params[2], oob_score = True)
    clf.fit(X_train, y_train)
    acc = []
    for X,y in data:
        acc.append(clf.score(X,y))

    lines = ['Training Accuracy is: ', 'Validation Accuracy is: ', 'Test Accuracy is: ']
    lines = [lines[i] + str(np.round(acc[i]*100,5)) + '\n' for i in range(len(lines))]
    
    lines.append('Out of bag Accuracy: ' + str(np.round(clf.oob_score_*100,5)) + '\n')
    lines = ["Results for Part d:\n"] + lines + ["\n"]
    return lines, acc

def best_part_e_params(X,y):
    estimator = XGBClassifier(objective= 'multi:softmax', nthread=4,seed=42)
    l = [0.4, 0.5, 0.6, 0.7, 0.8]
    param_dict = {"n_estimators": range(50,450,50), 
                     "subsample": l, 
                     "max_depth": range(40,70,10)
                     }

    grid = HalvingGridSearchCV(estimator=estimator, param_grid=param_dict, n_jobs = 10, cv = 5, verbose=True)
    grid.fit(X,y)
    p = grid.best_params_

    return p["n_estimators"], p["subsample"], p["max_depth"]

def part_e(data, output_path):
    X_train, y_train = data[0]
    params = best_part_e_params(X_train, y_train)
    clf = XGBClassifier(objective= 'multi:softmax', nthread=4,seed=42, n_estimators = params[0], subsample = params[1], max_depth = params[2])
    clf.fit(X_train, y_train)
    acc = []
    for X,y in data:
        acc.append(clf.score(X,y))
    
    lines = ['Training Accuracy is: ', 'Validation Accuracy is: ', 'Test Accuracy is: ']
    lines = [lines[i] + str(np.round(acc[i]*100,5)) + '\n' for i in range(len(lines))]
    
    return lines, acc

def part_f(data, output_path):
    X_train, y_train = data[0]
    X_validation, y_validation = data[1]
    X_test, y_test = data[2]
    """model = lgb.LGBMClassifier(learning_rate = 0.01, max_depth = -1, num_leaves = 100, max_bin = 511, random_state = 42)
    model.fit(X_train, y_train, eval_metric = 'logloss')
    lines = []
    lines.append("Training Accuracy is {:.4f}".format(model.score(X_train, y_train)))
    lines.append("Validation Accuracy is {:.4f}".format(model.score(X_validation, y_validation)))
    lines.append("Test Accuracy is {:.4f}".format(model.score(X_test, y_test)))"""
    return None, None

def plot_g(arr, output_path, part):
    sizes = [20000, 40000, 60000, 80000, 10000]
    test = []
    times = []
    for i in range(len(arr)):
        test.append(arr[i][0])
        times.append(arr[i][1])
    plt.title("Test Accuracy vs Sample Size for Part " + part)
    plt.xlabel("Sample Size")
    plt.ylabel("Test Accuracy")
    plt.plot(sizes, test)
    plt.savefig(output_path + '/Part_g_Acc_' + part)
    plt.clf()
    
    plt.title("Training Time vs Sample Size for Part " + part)
    plt.xlabel("Sample Size")
    plt.ylabel("Training Time")
    plt.plot(sizes, times)
    plt.savefig(output_path + '/Part_g_Time_' + part)
    plt.clf()

def part_g(data, output_path):
    X_train, y_train = data[0]
    X_validation, y_validation = data[1]
    X_test, y_test = data[2]
    sizes = [20000, 40000, 60000, 80000, 10000]
    a = []
    b = []
    c = []
    d = []
    e = []
    f = []
    
    for i in range(len(sizes)):
        idx = np.random.randint(112908, size=sizes[i])
        train1 = X_train[idx, :]
        train2 = y_train[idx, :]
        data[0] = train1, train2
        begin = time.time()
        lines, acc = part_a(data, output_path)
        end = time.time() - begin
        a.append([acc[2], end])
        
        begin = time.time()
        lines, acc = part_b(data, output_path)
        end = time.time() - begin
        b.append([acc[2], end])
        
        begin = time.time()
        lines, acc = part_c(data, output_path)
        end = time.time() - begin
        c.append([acc[2], end])
        
        begin = time.time()
        lines, acc = part_d(data, output_path)
        end = time.time() - begin
        d.append([acc[2], end])
        
        begin = time.time()
        lines, acc = part_e(data, output_path)
        end = time.time() - begin
        e.append([acc[2], end])
        
        begin = time.time()
        lines, acc = part_f(data, output_path)
        end = time.time() - begin
        f.append([acc[2], end])
        
    plot_g(a, output_path, 'a')
    plot_g(b, output_path, 'b')
    plot_g(c, output_path, 'c')
    plot_g(d, output_path, 'd')
    plot_g(e, output_path, 'e')
    plot_g(f, output_path, 'f')
    
    return None, None

train_data_path = sys.argv[1]
validation_data_path = sys.argv[2]
test_data_path = sys.argv[3]
output_path = sys.argv[4]
part = sys.argv[5]
paths = (train_data_path, validation_data_path, test_data_path)

data = get_data(paths)
begin = time.time()
part_dict = {'a':part_a, 'b':part_b, 'c': part_c_new, 'd': part_d, 'e':part_e, 'f':part_f, 'g':part_g}
result, acc = part_dict[part](data, output_path)
file = open(output_path + '/2_' + part + ".txt", "w")
file.writelines(result)
file.close()