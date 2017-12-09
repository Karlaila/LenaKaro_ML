import collections
import matplotlib.pyplot as plt
import data_visualizer as dv
import utilities as ut



features, labels = ut.parse(data_limit=-1)
test_features = ut.parse(path_feature='test_data.csv', path_labels='', data_limit = -1)
print 'feature number ' +str(len(test_features))


"""Lena's"""
Y_pred_rc = ut.do_rc(test_features, features, labels)
Y_pred_nn, Y_prob_nn = ut.do_nn(test_features, features, labels)

ut.output_labels(Y_pred_rc, filename='labels_rc')
ut.output_labels(Y_pred_nn, filename='labels_nn')
ut.output_proba(Y_prob_nn, filename='proba_nn')
