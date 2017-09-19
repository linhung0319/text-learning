#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

def plot_accuracy_score(score, C_range, gamma_range, kernel_range):
	titles = ['{} SVM Accuracy Score'.format(x) for x in kernel_range] 

	fig, axes = plt.subplots(nrows=1, ncols=len(kernel_range))
	for index, (ax, title) in enumerate(zip(axes, titles)):
		im = ax.imshow(score[:, :, index], vmin=0.5, vmax=1)
		
		ax.set_title(title)
		
		ax.set_xlabel('gamma')
		ax.set_xticks(np.arange(len(gamma_range)))
		ax.set_xticklabels(gamma_range, rotation=45)
		
		ax.set_ylabel('C')
		ax.set_yticks(np.arange(len(C_range)))
		ax.set_yticklabels(C_range)

	plt.tight_layout()
	plt.colorbar(im, ax=axes.ravel().tolist())
		
	plt.show()
