import csv
import collections
import matplotlib.pyplot as plt
from sklearn import svm
from random import shuffle
import data_visualizer as dv

# X id, 2-49 MFCCs, 50-97 Chroma, 98-265 Rhytm

def parse(path_feature='train_data.csv', path_labels='train_labels.csv', data_limit = -1):
    X = []
    with open(path_feature, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        counter = 0
        for row in reader:
            if counter > 0:
                X.append([float(i) for i in row])
            if data_limit > 0:
                if counter > data_limit:
                    break
            counter+=1

    counter = 0
    Y = []
    with open(path_labels, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if counter > 0:
                Y.append(float(row[0]))
            if data_limit > 0:
                if counter > data_limit:
                    break
            counter+=1

    return X, Y

def divide_set(features, labels):
    X_test, Y_test, X_train, Y_train = [], [], [], []

    # dividing into test and train data
    for i in range(0, len(labels)):
        if i % 4 == 0:
            X_test.append(features[i][50:265])
            Y_test.append(labels[i])
        else:
            X_train.append(features[i][50:265])
            Y_train.append(labels[i])
    return X_test, Y_test, X_train, Y_train

def do_svc(X_test, Y_test, X_train, Y_train):
    clf = svm.SVC(decision_function_shape='ovo', class_weight='balanced')
    print "starts fitting"
    print clf.fit(X_train, Y_train)

    print "finished fitting, starts predicting"
    ok = 0
    notok = 0
    Y_pred = []
    for i in range(0, len(Y_test)):
        result = clf.predict([X_test[i]])
        Y_pred.append(result)
        if result == Y_test[i]:
            ok += 1
        else:
            notok += 1
        #if i < 900:
        #    print result, Y_test[i]

    dv.confusionMatrix(Y_test, Y_pred)
    return "ok: ", ok, "; not ok:", notok, "; percentage: ", (ok*100)/(notok+ok)


features, labels = parse(data_limit=-1)

X_test, Y_test, X_train, Y_train = divide_set(features, labels)
print len(X_test), len(Y_test), len(Y_train)

#print "features \n", features[1:10]
#print "labels \n", labels[1:10]

"""showing the distribution of classes"""
"""counter = collections.Counter(labels)
print(counter.values())
print(counter.keys())
print(counter.most_common(3))

width = 1 / 1.5
plt.bar(counter.keys(), counter.values(), width, color="blue")
plt.show()"""""

"""SVC classification"""
print do_svc(X_test, Y_test, X_train, Y_train)