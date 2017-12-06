import csv
import collections
import matplotlib.pyplot as plt
from sklearn import svm
from random import shuffle


def parse(path_feature='train_data.csv', path_labels='train_labels.csv', data_limit = -1):
    X = []
    with open(path_feature, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        counter = 0
        for row in reader:
            X.append([float(i) for i in row])
            if data_limit > 0:
                if counter > data_limit:
                    break
                else:
                    counter+=1
    counter = 0
    Y = []
    with open(path_labels, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            Y.append(float(row[0]))
            if data_limit > 0:
                if counter > data_limit:
                    break
                else:
                    counter+=1

    return X, Y

def divide_set(features, labels):
    X_test, Y_test, X_train, Y_train = [], [], [], []

    # dividing into test and train data
    for i in range(0, len(labels)):
        if i % 4 == 0:
            X_test.append(features[i])
            Y_test.append(labels[i])
        else:
            X_train.append(features[i])
            Y_test.append(labels[i])
    return X_test, Y_test, X_train, Y_train


features, labels = parse(data_limit=-1)

X_test, Y_test, X_train, Y_train = divide_set(features, labels)

#print "features \n", features[1:10]
#print "labels \n", labels[1:10]

"""showing the distribution of classes"""
counter = collections.Counter(labels)
print(counter.values())
print(counter.keys())
print(counter.most_common(3))

width = 1 / 1.5
plt.bar(counter.keys(), counter.values(), width, color="blue")
plt.show()

"""SVC classification"""
clf = svm.SVC(decision_function_shape='ovo')
print "starts fitting"
print clf.fit(X_train, Y_train)

print "finished fitting, starts predicting"
ok = 0
notok = 0
for i in range(0, len(Y_test)):


#dec = clf.decision_function()

# [(1, 4), (2, 4), (3, 2)]