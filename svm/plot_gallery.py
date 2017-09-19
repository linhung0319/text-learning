#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

def plot_accuracy_score(score, C_range, gamma_range, kernel_range):
	"""
	    Plot the accuracy score of SVM across different C, gamma, kernel parameter
	    Input 3D array score['C']['gamma']['kernel']

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
