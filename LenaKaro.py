import collections
import matplotlib.pyplot as plt
import data_visualizer as dv
import utilities as ut



features, labels = ut.parse(data_limit=-1)
test_features = ut.parse(path_feature='test_data.csv', path_labels='', data_limit = -1)
print 'featur number ' +str(len(test_features))

#X_test, Y_test, X_train, Y_train = ut.divide_set(features, labels)
#print len(X_test), len(Y_test), len(Y_train)

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


"""classification"""
#Y_pred = ut.do_nn(X_test, Y_test, X_train, Y_train)
#print ut.check(Y_test, Y_pred)
#dv.confusionMatrix(Y_test, Y_pred)

"""logloss classification"""

Y_predLL = ut.do_mlp(X_test, Y_test, X_train, Y_train)
print "Alg res: ", ut.checkLogLoss(Y_test, Y_predLL)
print "Dummy res: ", ut.ckeckLogLossDummy(Y_test)

"""Lena's"""
#Y_pred = ut.do_rc_submit(test_features, features, labels)
#ut.output_labels(Y_pred, filename='labels_rc')
#ut.do_rc(X_test, Y_test, X_train, Y_train)
#print ut.check(Y_test, Y_pred)
#dv.confusionMatrix(Y_test, Y_pred, normalize=True)
