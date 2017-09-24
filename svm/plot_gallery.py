#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

def plot_accuracy_score(score, C_range, gamma_range, kernel_range):
	"""
	    Plot the accuracy score of SVM across different C, gamma, kernel parameter

	    score - Input 3D array score['C']['gamma']['kernel']

	"""
	titles = ['{} SVM Accuracy Score'.format(x) for x in kernel_range] 
	
	fig, axes = plt.subplots(nrows=1, ncols=len(kernel_range), figsize=(10, 5))

	for index, (ax, title) in enumerate(zip(axes, titles)):
		im = ax.imshow(score[:, :, index], vmin=0.5, vmax=1, interpolation='nearest')
		
		ax.set_title(title, fontweight='bold', size=15)
		
		ax.set_xlabel('gamma', fontsize=20)
		ax.set_xticks(np.arange(len(gamma_range)))
		ax.set_xticklabels(['%.1g' %x for x in gamma_range], rotation=45)
		
		ax.set_ylabel('C', fontsize=20, rotation=90)
		ax.set_yticks(np.arange(len(C_range)))
		ax.set_yticklabels(['%.1g' %y for y in C_range])
		
	fig.tight_layout(w_pad=2)
	fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7)
		
	plt.savefig('C gamma Accuracy Score.png')

def plot_feature_importance(coef, features_name, top_features=20):
	"""
	    Plot the top feature importance in SVM binary classification result

	    coef - weights assigned to the features (coefficients of the support vector in SVM decision function)
	    features_name - the feature's name
	    top_features - the amount of the feature (each class) plotted in the figure
	"""
	sorted_coef = np.argsort(coef)
	positive_top_coef = sorted_coef[-top_features:]
	negative_top_coef = sorted_coef[:top_features]
	top_coef = np.hstack([negative_top_coef, positive_top_coef])

	fig = plt.figure(figsize=(15, 7))
	
	plt.bar(np.arange(top_features),
		coef[negative_top_coef],
		color='red',
		align='edge',
		label='Sara')
	plt.bar(np.arange(top_features, 2 * top_features),
		coef[positive_top_coef],
		color='blue',
		align='edge',
		label='Chris')
	
	plt.title('Feature Importance', fontweight='bold', size=25)

	bins = np.arange(2 * top_features)
	plt.xlabel('Feature Name', fontsize=25)
	plt.xticks(bins, features_name[top_coef], rotation=60)
	plt.xlim([0, bins.size])

	plt.ylabel('Weighting', fontsize=20)

	plt.legend(loc='lower right', prop={'size': 30})
	plt.tight_layout()
	plt.savefig('Feature Importance.png')
