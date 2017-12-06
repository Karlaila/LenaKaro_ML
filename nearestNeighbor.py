import LenaKaro as lk
from sklearn import neighbors, datasets
#from sklearn.neighbors import NearestNeighbors
import numpy as np
import data_visualizer as dv

n_neighbors = 15
h = .02

X, Y = lk.parse()
X_test, Y_test, X_train, Y_train = lk.divide_set(X, Y)


clf = neighbors.KNeighborsClassifier(n_neighbors)
print 'fitting nearest neighbors'
clf.fit(X_train, Y_train)

print 'predicting'
Y_pred = clf.predict(X_test)

dv.confusionMatrix(Y_test, Y_pred, True)
