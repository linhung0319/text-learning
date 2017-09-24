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
from plot_gallery import plot_accuracy_score, plot_feature_importance

import matplotlib.pyplot as plt

def main():
	""" 
	    Use a SVM to identify emails from the Enron corpus by their authors:    
	    Sara has label 0
	    Chris has label 1
	"""
	### features_train and features_test are the features for the training
	### and testing datasets, respectively
	### labels_train and labels_test are the corresponding item labels
	features_train, features_test, labels_train, labels_test, features_name = preprocess()

	### scaled features and labels are 1/10 training data for tuning the 
	### parameter in SVM
	scaled_features = features_train[:int(len(features_train) / 10)]
	scaled_labels = labels_train[:int(len(labels_train) / 10)]

	### find the best parameter (kernel, C, gamma) in SVM
	param = find_best_param(scaled_features,
			        scaled_labels,
				scoring=make_scorer(accuracy_score),
				C_range=np.logspace(-2, 10, 13),
				gamma_range=np.logspace(-9, 3, 13),
				kernel_range=('linear', 'rbf'),
				kernel='linear')

	### linear SVM to fit the training data
	clf = SVC(**param)
	clf.fit(features_train, labels_train)
	
	### plot the top feature importance of the SVM binary classification result
	plot_feature_importance(clf.coef_.ravel(),
			        features_name,
			        top_features=20)
	
	
def find_best_param(features, labels, scoring, C_range, gamma_range, kernel_range, kernel=None):
	"""
	    Find the best parameter in SVM

	    scoring - the method to determine the accuracy in SVM
	    C_range, gamma_range - tuning parameters in SVM
	    kernel_range - different kernel in SVM, ex: ['linear', 'rbf', 'poly']

	    kernel - choose the best parameter in a given kernel
	    
	    if kerel = None, this function will return the best parameter across all kernel_range

	    1 dict object is returned:
	    	-- {'C', 'gamma', 'kernel'}
	"""
	logger = logging.getLogger(find_best_param.__name__)
	
	param_grid = {'C': C_range,
		      'gamma': gamma_range,
		      'kernel': kernel_range}

	clf = GridSearchCV(SVC(), param_grid, scoring)
	clf.fit(features, labels)
	### score['C']['gamma']['kernel']
	score = clf.cv_results_['mean_test_score'].reshape(len(C_range), len(gamma_range), len(kernel_range))
	
	### plot the accuracy score 
	plot_accuracy_score(score, C_range, gamma_range, kernel_range)
	
	if kernel == None:
		return clf.best_params_
	elif kernel not in kernel_range:
		logger.info('Your chosen kernel is not in kernel range, return the best parameter')
		return clf.best_params_
	else:
		kernel_index = kernel_range.index(kernel)	
		index = np.unravel_index(score[:, :, kernel_index].argmax(), score[:, :, kernel_index].shape)
		return {'C': C_range[index[0]], 'gamma': gamma_range[index[1]], 'kernel': kernel}

if __name__ == '__main__':
	main()
