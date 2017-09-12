#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

def plot_accuracy_score(score, C_range, gamma_range, kernel_range):
	titles = ['Validation Accuracy', 'Validation Accuracy']
	fig, axes = plt.subplots(nrows=1, ncols=len(kernel_range))
	for ax, title, i in zip(axes ,titles ,range(len(kernel_range))):
		im = ax.imshow(score[:][:][i], vmin=0.6, vmax=1.0)
		ax.set_title(title)
		ax.set_xlabel('C')
		ax.set_ylabel('gamma')
		ax.set_xticks(np.arange(len(C_range)))
		ax.set_xticklabels(C_range, rotation=45)
		ax.set_yticks(np.arange(len(gamma_range)))
		ax.set_yticklabels(gamma_range)

	fig.tight_layout()
	plt.colorbar(im, ax=axes.ravel().tolist(), shrink=0.5)
	plt.show()
