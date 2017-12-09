import parser
import csv
import collections
import matplotlib.pyplot as plt

import itertools
import numpy as np

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

names = ['Pop_Rock', 'Electronic', 'Rap', 'Jazz', 'Latin', 'RnB', 'International', 'Country', 'Reggae', 'Blues']

def visualizeLabels(labels = -1):
	if labels == -1:
		features, labels = parser.parse()

	counter=collections.Counter(labels)
	print(counter.values())
	print(counter.keys())
	print(counter.most_common(3))

	width = 1/1.5
	plt.bar(counter.keys(), counter.values() , width, color="blue")
	plt.show()


def confusionMatrix(y_test, y_pred,normalize=False, classes = names):
	print('starting to plotting function')
	cmap=plt.cm.Blues
	print('creating confusion matrix')
	cm = confusion_matrix(y_test, y_pred)
	title = 'confusion matrix'

	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix, without normalization')

	print(cm)
	plt.figure()
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.show()

	

#visualizeLabels()
#confusionMatrix([1,2,3,4,5,3,2,1,5,3,5,4,], [1,2,3,4,1,3,2,1,1,2,1,4], names)
