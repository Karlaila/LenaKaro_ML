import csv
import collections
import matplotlib.pyplot as plt

def parse(path_feature = 'train_data.csv', path_labels = 'train_labels.csv'):
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
            
    return X,Y
# [(1, 4), (2, 4), (3, 2)]d