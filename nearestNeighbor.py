import parser
from sklearn import svm

features, labels = parser.parse()
clf = svm.SVC(decision_function_shape='ovo')
