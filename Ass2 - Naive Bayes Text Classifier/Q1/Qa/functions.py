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

def get_vocab(words_matrix):
    vocab = set()
    for words in words_matrix:
        for word in words:
            if len(word)>0:
                vocab.add(word.lower())

    train_vocab = list(vocab)
    vocab_dict = {}
    for i in range(len(train_vocab)):
        vocab_dict[train_vocab[i]] = i
        
    return vocab_dict

def get_words_matrix(path, y, label, removal):
    words_matrix = []
    words_str = []
    dir_list = os.listdir(path)
    m = len(dir_list)
    nlp = English()
    for i in range(len(dir_list)):
        with open(path + '/' + dir_list[i]) as f:
            content = process(f.readlines()[0], removal, nlp)
            words_str.append(content)
            words_matrix.append(list(set(content.split())))
            y.append(label)
    return m,y, words_matrix, words_str

def get_data(path, removal):
    y = []
    m1, y, pos, words_pos = get_words_matrix(path + 'pos', y, 1, removal)
    m0, y, neg, words_neg = get_words_matrix(path + 'neg', y, 0, removal)
    m = len(y)
    y = np.array(y).reshape(m,1)
    return y, pos+neg, m1/(m1+m0), words_pos, words_neg
    
def get_params(vocab, words_matrix, y):
    n = len(vocab)
    phi_1 = np.ones(n)
    phi_0 = np.ones(n)
    dict_set = set(vocab.keys())
    total_1 = 0
    total_0 = 0
    for i in range(len(words_matrix)):
        words = words_matrix[i]
        if y[i][0]==1:
            total_1+=len(words)
        if y[i][0]==0:
            total_0+=len(words)
        for word in words:
            if len(word)>0:
                if y[i][0]==1:
                    phi_1[vocab[word.lower()]]+=1
                if y[i][0]==0:
                    phi_0[vocab[word.lower()]]+=1
    phi_1 = phi_1/(total_1+n)
    phi_0 = phi_0/(total_0+n)
    return phi_1, phi_0

def predict(py, phi_1, phi_0,words, vocab, skip_check):
    prod = py/(1-py)
    vocab_set = set(vocab)
    for word in words:
        if len(word)>0 and (skip_check or word in vocab_set):
            prod*=phi_1[vocab[word.lower()]]/phi_0[vocab[word.lower()]]
    if prod>=1:
        return 1
    else:
        return 0

def get_accuracy(words_matrix, py, phi_1, phi_0, vocab, y, skip_check):
    y_pred = np.zeros(y.shape)
    for i in range(len(words_matrix)):
        y_pred[i] = predict(py, phi_1, phi_0, words_matrix[i], vocab, skip_check)
    print(np.round((y_pred==y).sum()*100/y.shape[0], 2))
    return y_pred

def get_wordcloud(pos, neg):
    pos_str = ' '.join(pos)
    neg_str = ' '.join(neg)
    
    specialchars = "-*.,;!><""/? @&{}|\~`#$%()^_+=:"
    pos_str = pos_str.replace("<br />", " ")
    neg_str = neg_str.replace("<br />", " ")
    for sp in specialchars:
        pos_str = pos_str.replace(sp, " ")
        neg_str = neg_str.replace(sp, " ")
        
    word_cloud = WordCloud(collocations = False, background_color = 'white').generate(pos_str)
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig('../plots/wordcloud1.jpg')
    
    word_cloud = WordCloud(collocations = False, background_color = 'white').generate(neg_str)
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig('../plots/wordcloud2.jpg')
