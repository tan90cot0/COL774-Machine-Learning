import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import GridSearchCV
import graphviz
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

def get_data_ignore(data):
    y = data["Severity"]
    X = data[["Age", "Shape", "Margin", "Density"]]
    X = X.to_numpy()
    to_remove = set()
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if X[i][j]=="?":
                to_remove.add(i)
            else:
                X[i][j] = int(X[i][j])
    X = np.delete(X, list(to_remove), axis = 0)
    y = y.to_numpy()
    y = np.delete(y, list(to_remove), axis = 0)
    return X,y

def get_data_impute(data, train, imp, func):
    y = data["Severity"]
    X = data[["Age", "Shape", "Margin", "Density"]]
    X = X.to_numpy()
    y = y.to_numpy()
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if X[i][j]=="?":
                X[i][j] = np.nan
            else:
                X[i][j] = int(X[i][j])
    if train:
        imp = SimpleImputer(missing_values=np.nan, strategy=func)
        imp.fit(X)
    X = imp.transform(X)

    return X,y, imp

def get_data_xgboost(data):
    y = data["Severity"]
    X = data[["Age", "Shape", "Margin", "Density"]]
    X = X.to_numpy()
    y = y.to_numpy()
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if X[i][j]=="?":
                X[i][j] = np.nan
            else:
                X[i][j] = int(X[i][j])
    return X,y

def get_data(paths, part, func):
    train_data_path, validation_data_path, test_data_path = paths
    
    train_data = pd.read_csv(train_data_path)
    validation_data = pd.read_csv(validation_data_path)
    test_data = pd.read_csv(test_data_path)

    if part in ['a', 'b', 'c', 'd']:
        X_train,y_train = get_data_ignore(train_data)
        X_validation,y_validation = get_data_ignore(validation_data)
        X_test,y_test = get_data_ignore(test_data)   
    elif part=='e':
        X_train,y_train, imp = get_data_impute(train_data, True, None, func)
        X_validation,y_validation, imp = get_data_impute(validation_data, False, imp, func)
        X_test,y_test, imp = get_data_impute(test_data, False, imp, func)
    else:
        X_train,y_train = get_data_xgboost(train_data)
        X_validation,y_validation = get_data_xgboost(validation_data)
        X_test,y_test = get_data_xgboost(test_data)

    train_data = (X_train, y_train)
    validation_data = (X_validation, y_validation)
    test_data = (X_test, y_test)
    
    return train_data, validation_data, test_data

def predict(clf, X,y):
    m,n = X.shape
    y_pred = np.zeros(m)
    for i in range(y.shape[0]):
        y_pred[i] = clf.predict(X[i].reshape(1,n))
    acc = (y_pred==y).sum()/m
    return acc

def visualise(clf, name):
    dot_data = tree.export_graphviz(clf, out_file=None, filled = True) 
    graph = graphviz.Source(dot_data, format = "png") 
    graph.render(name) 
    
def plot_alpha_graphs(alphas, metrics, scores, output_path):
    train_scores, validation_scores, test_scores = scores
    plt.xlabel("Alphas")
    plt.ylabel("Accuracies")
    plt.title("Accuracy vs Alpha for Training and Test sets")
    plt.plot(alphas, train_scores, marker="o", label="Train", drawstyle="steps-post")
    plt.plot(alphas, validation_scores, marker="o", label="Validation", drawstyle="steps-post")
    plt.plot(alphas, test_scores, marker="o", label="Test", drawstyle="steps-post")
    plt.legend()
    plt.savefig(output_path + '/Accuracy_1')
    plt.clf()

    impurities, nodes, depth = metrics
    
    plt.xlabel("Alphas")
    plt.ylabel("Impurities")
    plt.title("Impurity vs Alpha")
    plt.plot(alphas, impurities, marker="o", label="Impurities", drawstyle="steps-post")
    plt.legend()
    plt.savefig(output_path + '/Impurity_1')
    plt.clf()


    plt.xlabel("Alphas")
    plt.ylabel("Nodes")
    plt.title("Nodes vs Alpha")
    plt.plot(alphas, nodes, marker="o", label="Nodes", drawstyle="steps-post")
    plt.legend()
    plt.savefig(output_path + '/Nodes_1')
    plt.clf()

    plt.xlabel("Alphas")
    plt.ylabel("Depth")
    plt.title("Depth vs Alpha")
    plt.plot(alphas, depth, marker="o", label="Depth", drawstyle="steps-post")
    plt.legend()
    plt.savefig(output_path + '/Depth_1')
    plt.clf()

def get_scores(data, clfs):
    train_data, validation_data, test_data = data
    X_train, y_train = train_data
    X_validation, y_validation = validation_data
    X_test, y_test = test_data 
    train_scores = [clf.score(X_train, y_train) for clf in clfs]
    validation_scores = [clf.score(X_validation, y_validation) for clf in clfs]
    test_scores = [clf.score(X_test, y_test) for clf in clfs]
    
    return train_scores, validation_scores, test_scores

def model(clf, data):
    X_train,y_train = data[0]
    clf.fit(X_train, y_train)
    acc = []
    for X,y in data:
        acc.append(clf.score(X,y))
    return acc

def part_a(data, output_path):
    clf = tree.DecisionTreeClassifier(random_state = 0)
    acc = model(clf, data)
    lines = ['Training Accuracy is: ', 'Validation Accuracy is: ', 'Test Accuracy is: ']
    lines = [lines[i] + str(np.round(acc[i]*100,5)) + '\n' for i in range(len(lines))]
    lines = ["Results for Part a:\n"] + lines + ["\n"]
    return lines, clf
    
def get_part_b_params(train_data):
    clf = tree.DecisionTreeClassifier()
    param_dict = {"max_depth": range(1,5), 
                 "min_samples_split": range(2,100), 
                 "min_samples_leaf": range(1,5)
                 }

    grid = GridSearchCV(clf, param_grid = param_dict, cv = 10, verbose = 1, n_jobs = -1)
    X_train,y_train = train_data
    grid.fit(X_train, y_train)
    p = grid.best_params_
    return p['max_depth'], p['min_samples_leaf'], p['min_samples_split']

def part_b(data, output_path):
    params = get_part_b_params(data[0])
    clf = tree.DecisionTreeClassifier(random_state = 0, max_depth = params[0], min_samples_leaf = params[1], min_samples_split = params[2])
    acc = model(clf, data)
    lines = ['Training Accuracy is: ', 'Validation Accuracy is: ', 'Test Accuracy is: ']
    lines = [lines[i] + str(np.round(acc[i]*100,5)) + '\n' for i in range(len(lines))]
    lines = ["Results for Part b:\n"] + lines + ["\n"]
    return lines, clf

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

def part_c(data, output_path):
    ccp_alphas, clfs, metrics = get_part_c_params(data[0])
    scores = get_scores(data, clfs)

    plot_alpha_graphs(ccp_alphas, metrics, scores, output_path)
    index = np.argmax(np.array(scores[1]))
    best_alpha, best_tree = ccp_alphas[index], clfs[index]
    acc = scores[0][index], scores[1][index], scores[2][index]
    
    lines = ['Training Accuracy is: ', 'Validation Accuracy is: ', 'Test Accuracy is: ']
    lines = [lines[i] + str(np.round(acc[i]*100,5)) + '\n' for i in range(len(lines))]
    lines.append('Best Alpha = ' + str(np.round(best_alpha,5)) + '\n')
    lines = ["Results for Part c:\n"] + lines + ["\n"]
    return lines, best_tree

def get_part_d_params(X,y):
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    param_dict = {"n_estimators": range(1,15), 
                    "max_features": ["sqrt", "log2", None], 
                    "min_samples_split": range(2,3)
                    }
    #Tested up to 1000 n-estimators
    #10 min sample splits

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
    return lines, None

def part_e(paths, output_path):
    agg_func = ["Results for 'mean' aggregate function:\n", "Results for 'median' aggregate function:\n"]
    funcs = ['mean', 'median']
    result = []
    for i in range(2):
        result+= [agg_func[i]]
        data = get_data(paths, 'e', funcs[i])
        temp, clf = part_a(data, output_path)
        result+=temp
        temp, clf =part_b(data, output_path)
        result+=temp
        temp, clf =part_c(data, output_path)
        result+=temp
        temp, clf =part_d(data, output_path)
        result+=temp
    return result

def best_part_f_params(X,y):
    estimator = XGBClassifier(objective= 'binary:logistic', nthread=4,seed=42)
    param_dict = {"n_estimators": range(10,50,10), 
                     "subsample": [0.1, 0.2, 0.3, 0.4, 0.5], 
                     "max_depth": range(4,10,1)
                     }

    grid = GridSearchCV(estimator=estimator, param_grid=param_dict, scoring = 'roc_auc', n_jobs = 10, cv = 10, verbose=True)
    grid.fit(X,y)
    p = grid.best_params_
    return p["n_estimators"], p["subsample"], p["max_depth"]

def part_f(data):
    X_train, y_train = data[0]
    params = best_part_f_params(X_train, y_train)
    clf = XGBClassifier(objective= 'binary:logistic', nthread=4,seed=42, n_estimators = params[0], subsample = params[1], max_depth = params[2])
    clf.fit(X_train, y_train)
    acc = []
    for X,y in data:
        acc.append(clf.score(X,y))
    
    lines = ['Training Accuracy is: ', 'Validation Accuracy is: ', 'Test Accuracy is: ']
    lines = [lines[i] + str(np.round(acc[i]*100,5)) + '\n' for i in range(len(lines))]
    
    return lines