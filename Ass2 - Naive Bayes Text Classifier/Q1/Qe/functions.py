import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS
from nltk.stem import PorterStemmer
from wordcloud import WordCloud

def process(article, removal, nlp):
    specialchars = "-*.,;!><''""/? @&{}|\~`#$%()^_+=:"
    article = article.replace("<br />", " ")
    for sp in specialchars:
        article = article.replace(sp, " ") 
        
    if removal:
        ps = PorterStemmer()
        for word in nlp(article):
            if nlp.vocab[word.text].is_stop == True:
                article = article.replace(word.text, "")
            else:
                article = article.replace(word.text, ps.stem(word.text))
    return article

def get_words_matrix(path, y, label, removal):
    words_matrix = []
    dir_list = os.listdir(path)
    m = len(dir_list)
    nlp = English()
    for i in range(len(dir_list)):
        with open(path + '/' + dir_list[i]) as f:
            content = process(f.readlines()[0], removal, nlp)
            words_matrix.append(list(set(content.split())))
            y.append(label)
    return m,y, words_matrix

def get_data(path, removal):
    y = []
    m1, y, pos = get_words_matrix(path + 'pos', y, 1, removal)
    m0, y, neg = get_words_matrix(path + 'neg', y, 0, removal)
    m = len(y)
    y = np.array(y).reshape(m,1)
    return y, pos+neg, m1/m
    
def get_params(words_matrix, y, trigrams):
    vocab = set()
    for words in words_matrix:
        vocab.add(words[0].lower())
        for i in range(1,len(words)):
            vocab.add(words[i].lower())
            vocab.add((words[i-1] + " " + words[i]).lower())
        if trigrams:
            for i in range(2,len(words)):
                trigram = words[i-2] + " " + words[i-1] + " " + words[i]
                vocab.add(trigram.lower())

    train_vocab = list(vocab)
    vocab_dict = {}
    for i in range(len(train_vocab)):
        vocab_dict[train_vocab[i]] = i

    n = len(vocab_dict)
    phi_1 = np.ones(n)
    phi_0 = np.ones(n)
    total_1 = 0
    total_0 = 0
    for i in range(len(words_matrix)):
        positive= y[i][0]==1
        negative= y[i][0]==0
        words = words_matrix[i]
        if positive:
            total_1+=len(words)
        if negative:
            total_0+=len(words)
        for word in words:
            ind = vocab_dict[word.lower()]
            if positive:
                phi_1[ind]+=1
            if negative:
                phi_0[ind]+=1

        for j in range(1, len(words)):
            bigram = words[j-1] + ' ' + words[j]
            ind = vocab_dict[bigram.lower()]
            if positive:
                phi_1[ind]+=1
            if negative:
                phi_0[ind]+=1
        
        if trigrams:
            for j in range(2, len(words)):
                trigram = words[j-2] + " " + words[j-1] + " " + words[j]
                ind = vocab_dict[trigram.lower()]
                if positive:
                    phi_1[ind]+=1
                if negative:
                    phi_0[ind]+=1

    phi_1 = phi_1/(total_1+n)
    phi_0 = phi_0/(total_0+n)
    phi = phi_1/phi_0
    return phi, vocab_dict

def get_accuracy(x, py, phi, y):
    y_pred = np.zeros(y.shape)
    for i in range(len(x)):
        y_pred[i] = 1 if np.prod(phi[x[i]]) * py/(1-py) >=1 else 0
    print(np.round((y_pred==y).sum()*100/y.shape[0], 2))
    return y_pred

def get_indices(words_matrix, vocab, skip_check, trigrams):
    x = []
    vocab_set = set(vocab.keys())
    for words in words_matrix:
        indices = []
        if skip_check or words[0].lower() in vocab_set:
            indices.append(vocab[words[0].lower()])
        for i in range(1,len(words)):
            bigram = (words[i-1] + " " + words[i]).lower()
            if skip_check or bigram in vocab_set:
                indices.append(vocab[bigram.lower()])
            if skip_check or words[i].lower() in vocab_set:
                indices.append(vocab[words[i].lower()])

        if trigrams:
            for i in range(2,len(words)):
                trigram = words[i-2] + " " + words[i-1] + " " + words[i]
                if skip_check or trigram.lower() in vocab_set:
                    indices.append(vocab[trigram.lower()])
        indices = np.array(indices)
        x.append(indices)
    return x

def train_model(path, removal, trigrams):
    y, words_matrix,py = get_data(path, removal)
    phi, vocab= get_params(words_matrix, y, trigrams)
    x = get_indices(words_matrix, vocab, True, trigrams)
    y_pred = get_accuracy(x, py, phi, y)
    params = phi, vocab, py
    return y_pred, params

def test_model(path, removal, params, trigrams):
    phi, vocab, py = params
    y, words_matrix,py2 = get_data(path, removal)
    x = get_indices(words_matrix, vocab, False, trigrams)
    y_pred = get_accuracy(x, py, phi, y)
    return y_pred
