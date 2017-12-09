import csv
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn import tree
import collections
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn import neighbors
import numpy as np


""" parsing """
def parse(path_feature='train_data.csv', path_labels='train_labels.csv', data_limit = -1):
    X = []
    with open(path_feature, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        counter = 0
        for row in reader:
            if counter > data_limit or data_limit == -1:
                X.append([float(i) for i in row])
            counter+=1

    counter = 0   
    if path_labels != '':
        Y = []
        with open(path_labels, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if counter > data_limit or data_limit == -1:
                    Y.append(float(row[0]))
                counter+=1
        return X, Y
    return X

def get2ClassLabels(Y):
    for y in Y:
        if y != 1:
            y = 0

def divide_set(features, labels, fraction = 4):
    X_test, Y_test, X_train, Y_train = [], [], [], []

    # dividing into test and train data
    for i in range(0, len(labels)):
        if i % fraction == 0:
            X_test.append(features[i][1:265])
            Y_test.append(labels[i])
        else:
            X_train.append(features[i][1:265])
            Y_train.append(labels[i])
    return X_test, Y_test, X_train, Y_train

def even_classes(X, Y, number = -1):
    counter=collections.Counter(Y)
    distribution = counter.values()
    print distribution
    if number == -1:
        number = counter.least_common(1)
    sums_train = [0,0,0,0,0,0,0,0,0,0]
    X_new = [] 
    Y_new = []
    for i in range(len(Y)):
        sums_train[int(Y[i]-1)] += 1
        if sums_train[int(Y[i]-1)] < number:
            X_new.append(X[i])
            Y_new.append(Y[i])
    return X_new, Y_new
#TODO continue

def output_labels(Y, filename='labels'):
    print 'output file with labels'
    with open(filename+'.csv', 'w') as csvfile:
        fieldnames = ['Sample_id', 'Sample_label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in xrange(1,len(Y)):
            writer.writerow({'Sample_id': str(int(i)), 'Sample_label': str(int(Y[i]))})


"""checking"""

def check (Y_test, Y_pred):
    ok = 0
    notok = 0
    for i in range(0, len(Y_test)):
        if Y_pred[i] == Y_test[i]:
            ok += 1
        else:
            notok += 1
            # if i < 900:
            #    print result, Y_test[i]
    return "ok: ", ok, "; not ok:", notok, "; percentage: ", (ok * 100) / (notok + ok)

def checkLogLoss(Y_test, Y_predLL):
    N = len(Y_test)
    res = 0
    for i in range(N):

        # normalize
        sumn = sum(Y_predLL[i])
        Y_predLL[i] = [Y_predLL[i][j]/sumn for j in range(10)]

        for cl in range(1,11):
            # sum only the prob of the right class
            if Y_test[i] == cl:
                if Y_predLL[i][cl-1] == 0:
                    Y_predLL[i][cl - 1] = 0.0000000001
                res += np.log(Y_predLL[i][cl-1])
    res /= N
    return -res

def ckeckLogLossDummy(Y_test):
    N = len(Y_test)
    res = 0
    for i in range(N):
        for cl in range(1, 11):
            # sum only the prob of the right class
            if Y_test[i] == 1:
                res += np.log(0.999999)

    res /= N
    return -res



"""accuracy"""

# linear model - Ridge Classifier - 62 ;o | with balanced class 54 | NOT LOGLOSS
def do_rc(X_test, X_train, Y_train):
    # creating a classifier of loss function "hinge" and penalty function "l2"
    clf = RidgeClassifier()
    print "starts fitting"
    print clf.fit(X_train, Y_train)
    print "finished fitting, starts predictions"
    Y_pred = clf.predict(X_test)
    print "finished predictions"
    return Y_pred

# linear model - Ridge Classifier with Cross Valifation - 62 ;o | with balanced class 51
def do_rcv(X_test, X_train, Y_train):
    # creating a classifier of loss function "hinge" and penalty function "l2"
    clf = RidgeClassifierCV()
    print "starts fitting"
    print clf.fit(X_train, Y_train)
    print "finished fitting, starts predictions"
    Y_pred = clf.predict(X_test)
    print "finished predictions"
    return Y_pred

# for followitg features the best score | NOT LOGLOSS
def do_svc(X_test, X_train, Y_train):
    clf = svm.SVC(decision_function_shape='ovo', class_weight='balanced', kernel = "poly")
    print "starts fitting"
    print clf.fit(X_train, Y_train)
    print "finished fitting"
    Y_pred = clf.predict(X_test)
    return Y_pred



"""logloss"""

# multi-layer perceptron (MLP) algorithm that trains using Backpropagation. | LOGLOSS
def do_mlp(X_test, X_train, Y_train):
    clf = MLPClassifier(solver='adam', hidden_layer_sizes=(100,4))
    clf.out_activation_ = "Softmax"
    clf.n_outputs_ = 10
    print "starts fitting"
    print clf.fit(X_train, Y_train)
    print clf.classes_
    print clf.out_activation_
    print "finished fitting, starts predictions"
    Y_pred = clf.predict_proba(X_test)
    print "finished predictions"
    return Y_pred


# for followitg features the best score | LOGLOSS
def do_nn(X_test, X_train, Y_train):
    n_neighbors = 31
    clf = neighbors.KNeighborsClassifier(n_neighbors)
    print "starts fitting"
    print clf.fit(X_train, Y_train)
    print "finished fitting"
    Y_pred = clf.predict_proba(X_test)
    return Y_pred


""" SLOW, LOW, not working """

# very low score (around 30)
def do_dec_tree(X_test, Y_test, X_train, Y_train):
    # creating a classifier of loss function "hinge" and penalty function "l2"
    clf = tree.DecisionTreeClassifier()
    print "starts fitting"
    print clf.fit(X_train, Y_train)
    print "finished fitting, starts predictions"
    Y_pred = clf.predict(X_test)
    print "finished predictions"
    return Y_pred


# Gaussian process classification - very slow
def do_gpc(X_test, Y_test, X_train, Y_train):
    # creating a classifier of loss function "hinge" and penalty function "l2"
    clf = GaussianProcessClassifier(kernel=1.0 * RBF(length_scale=1.0))
    print "starts fitting"
    print clf.fit(X_train, Y_train)
    print "finished fitting, starts predictions"
    Y_pred = clf.predict(X_test)
    print "finished predictions"
    return Y_pred


# Gaussian Naive Bayes - fast;
def do_gnb(X_test, Y_test, X_train, Y_train):
    clf = GaussianNB()
    print "starts fitting"
    print clf.fit(X_train, Y_train)
    print "finished fitting, starts predictions"
    Y_pred = clf.predict(X_test)
    print "finished predictions"
    return Y_pred

# Multinomial Naive Bayes - cannot use this, as we have negative features
def do_mnb(X_test, Y_test, X_train, Y_train):
    clf = MultinomialNB()
    print "starts fitting"
    print clf.fit(X_train, Y_train)
    print "finished fitting, starts predictions"
    Y_pred = clf.predict(X_test)
    print "finished predictions"
    return Y_pred

# very random, no matter of loss and penalty functions, between 35-56 percent
def do_sgd(X_test, Y_test, X_train, Y_train):
    # creating a classifier of loss function "hinge" and penalty function "l2"
    clf = SGDClassifier(loss="hinge", penalty="l2")
    print "starts fitting"
    print clf.fit(X_train, Y_train)
    print "finished fitting, starts predictions"
    Y_pred = clf.predict(X_test)
    print "finished predictions"
    return Y_pred
