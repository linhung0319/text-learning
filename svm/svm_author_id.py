#!/usr/bin/env python3

import logging
logging.basicConfig(level=logging.INFO,
		    format='%(name)s - %(message)s')

import sys
sys.path.append('../preprocess')

import numpy as np

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score

from email_preprocess import preprocess
from plot_gallery import plot_accuracy_score

def main():
	""" 
	    Use a SVM to identify emails from the Enron corpus by their authors:    
	    Sara has label 0
	    Chris has label 1
	"""
	### features_train and features_test are the features for the training
	### and testing datasets, respectively
	### labels_train and labels_test are the corresponding item labels
	features_train, features_test, labels_train, labels_test = preprocess()

	### scaled features and labels are 1/10 training data for tuning the 
	### parameter in SVM
	scaled_features = features_train[:int(len(features_train) / 10)]
	scaled_labels = labels_train[:int(len(labels_train) / 10)]

	test_different_kernel_in_svm(scaled_features, scaled_labels)

def test_different_kernel_in_svm(features, labels):
	"""
	    scoring - the method to determine the accuracy in SVM
	    C_range, gamma_range - tuning parameters in SVM
	    kernel_range - different kernel in SVM, ex: 'linear', 'rbf', 'poly'
	"""	
	scoring = make_scorer(accuracy_score)
	C_range = np.logspace(-2, 10, 13)
	gamma_range = np.logspace(-9, 3, 13)
	kernel_range = ('linear', 'rbf')

	param_grid = {'C': C_range,
		      'gamma': gamma_range,
		      'kernel': kernel_range}

	clf = GridSearchCV(SVC(), param_grid, scoring)
	clf.fit(features, labels)
	### score['C']['gamma']['kernel']
	score = clf.cv_results_['mean_test_score'].reshape(len(C_range), len(gamma_range), len(kernel_range))
	
	plot_accuracy_score(score, C_range, gamma_range, kernel_range)

if __name__ == '__main__':
	main()
