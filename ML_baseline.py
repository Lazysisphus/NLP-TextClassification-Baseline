'''
@Descripttion: 
@version: 
@Author: Zhang Xiaozhu
@Date: 2019-06-01 20:03:34
@LastEditors  : Zhang Xiaozhu
@LastEditTime : 2019-12-23 13:39:41
'''


import re
import pandas as pd
import numpy as np

from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

'''
method
'''
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron

'''
metrics
'''
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score


train_df = pd.read_pickle('./data/mid_data/full/full_train.pkl')
dev_df = pd.read_pickle('./data/mid_data/full/full_dev.pkl')
test_df = pd.read_pickle('./data/mid_data/full/full_test.pkl')


if __name__ == '__main__':
    '''
    get data
    '''
    tfidf_word = TfidfVectorizer(sublinear_tf=True, min_df=3, analyzer='word', ngram_range=(1, 2))
    sen_cut_list = []
    for i in range(len(train_df)):
        sent = ' '.join(train_df.sen_cut[i])
        sen_cut_list.append(sent)
    for i in range(len(test_df)):
        sent = ' '.join(test_df.sen_cut[i])
        sen_cut_list.append(sent)

    train_x = tfidf_word.fit_transform(sen_cut_list)[:len(train_df)].toarray()
    test_x = tfidf_word.fit_transform(sen_cut_list)[len(train_df):].toarray()

    '''
    get label
    '''
    train_y = np.array(train_df['label'].tolist())
    test_y = np.array(test_df['label'].tolist())

    '''
    chose model
    '''
    # clf = LogisticRegression(C=1.0)
    clf = SVC(kernel='linear', C=1.0)
    # clf = GaussianNB()
    # clf = Perceptron(fit_intercept=False, max_iter=30, shuffle=False)
    
    '''
    train
    '''
    clf.fit(train_x, train_y)

    '''
    predict
    '''
    pred = clf.predict(test_x)
    f1 = f1_score(test_y, pred)
    precision = precision_score(test_y, pred)
    recall = recall_score(test_y, pred)
    acc = accuracy_score(test_y, pred)
    
    print('precision: {:.4f}, recall: {:.4f}, f1: {:.4f}, acc: {:.4f}'.format(precision, recall, f1, acc))

    '''
    result record
    model         base     拼音    情感   句子长度   词性    否定词   重复词    括号    全部特征
    LR :         32.64    42.24   32.53    33.02   35.24   33.07    33.04   41.09    50.57
    SVM:         39.71    45.20   39.53    39.28   42.53   40.29    39.60   45.36    51.49
    perceptron:  45.23    48.94   45.58    43.82   44.40   37.01    42.91   49.18    49.26
    '''
