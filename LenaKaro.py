import csv
import collections
import matplotlib.pyplot as plt
from sklearn import svm


def parse(path_feature='train_data.csv', path_labels='train_labels.csv'):
    X = []
    with open(path_feature, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            X.append([float(i) for i in row])

    Y = []
    with open(path_labels, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            Y.append(float(row[0]))

    return X, Y


features, labels = parse()

counter = collections.Counter(labels)
print(counter.values())
print(counter.keys())
print(counter.most_common(3))

width = 1 / 1.5
plt.bar(counter.keys(), counter.values(), width, color="blue")
plt.show()
# [(1, 4), (2, 4), (3, 2)]