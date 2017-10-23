import sys
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import preprocessing

if __name__ == "__main__":
	train_file = open("../data/row/train-full.txt")
	dev_file = open("../data/row/dev-full.txt")
	train_file.readline()
	dev_file.readline()
	train_vec  = np.load("../data/feature/train_vec.npy")
	train_labels = np.load("../data/feature/train_labels.npy")
	dev_vec = np.load("../data/feature/dev_vec.npy")
	dev_labels = np.load("../data/feature/dev_lables.npy")
	train_vec = np.nan_to_num(train_vec)
	dev_vec = np.nan_to_num(dev_vec)
	train_vec = train_vec[:,2:9]
	dev_vec = dev_vec[:,2:9]
	train_vec = preprocessing.scale(train_vec)
	train_labels = map(float, train_labels)
	dev_vec = preprocessing.scale(dev_vec)
	#clf = LogisticRegression(penalty='l1')
	clf = LogisticRegression()
	clf.fit(train_vec, train_labels)
	pred = []
	for i in range(0, len(dev_vec), 2):
		item1 = clf.predict_proba([dev_vec[i]])
		item2 = clf.predict_proba([dev_vec[i+1]])
		if item1[0][1] > item2[0][1]:
			pred.append(0)
		else:
			pred.append(1)
	acc = accuracy_score(pred, dev_labels)
	print acc
