import utilities as lk
from sklearn import neighbors, datasets
#from sklearn.neighbors import NearestNeighbors
import numpy as np
import data_visualizer as dv
from sklearn.metrics import accuracy_score

n_neighbors = 31
h = .02

X, Y = lk.parse(data_limit = 1500)
X_test, Y_test, X_train, Y_train = lk.divide_set(X, Y)


clf = neighbors.KNeighborsClassifier(n_neighbors)
print 'fitting nearest neighbors'
clf.fit(X_train, Y_train)

print 'predicting'
Y_pred = clf.predict(X_test)

print(accuracy_score(Y_test, Y_pred))
dv.confusionMatrix(Y_test, Y_pred, True)

