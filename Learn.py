#####
# LEARN: ML CLASS
#####

import os
import time

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB

def learn(train_data, test_data, mod="SVM", keys=False, param=None, tag="None"):
    start = time.time()
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    test_keys = []

    for key in train_data:
        X_train.append([float(x) for x in train_data[key][0:-1]])
        y_train.append(float(train_data[key][-1]))
    for key in test_data:
        test_keys.append(key)
        X_test.append([float(x) for x in test_data[key][0:-1]])
        y_test.append(float(test_data[key][-1]))

#    np.array(X_train, float)
#    np.array(y_train, float)

#    np.array(X_test, float)
#    np.array(y_test, float)

    correct_ones = []
    false_negs = []
    false_poss = []

    max_f1 = 0
    if mod == "LR":
        model = LogisticRegression(C=1) 
    elif mod == "BNB":
        model = BernoulliNB(alpha=param, binarize=0.0)
    elif mod == "SVM":
        model = SVC(C=param)

    model.fit(X_train, y_train)
    y_preds = model.predict(X_test)
        
    #if you want to return the actual predictions made, say if running on real unlabelled data
    if keys:
        res = {}
        for i in range(len(y_preds)):
            res[test_keys[i]] = y_preds[i]
        return (res, evaluate(y_test, y_preds))
    else:
        return evaluate(y_test, y_preds)

def evaluate(y_gold, y_preds):
    false_pos = 0
    false_neg = 0
    cor = 0
    total = len(y_gold)

    for i in range(len(y_gold)):
        if int(y_gold[i]) == 1 and int(y_preds[i]) == 0:
            false_neg += 1
        elif int(y_gold[i]) == 0 and int(y_preds[i]) == 1:
            false_pos += 1
        elif int(y_gold[i]) == 1 and int(y_preds[i]) == 1:
            cor += 1

    if (cor + false_pos) == 0:
        prec = 0.
    else:
        prec = float(cor) / (cor + false_pos)

    if (cor + false_neg) == 0:
        rec = 0.
    else:
        rec = float(cor) / (cor + false_neg)
        
    if prec+rec == 0:
        f1 = 0.
    else:
        f1 =2*(prec*rec)/(prec+rec)

    return f1, prec, rec
