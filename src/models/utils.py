#utils_unet

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn




def dice_loss(pred, target, smooth = 1.):
	pred = pred.contigous()
	target = target.contigous()
	
	intersection = (pred * target).sum(dim=2).sum(dim=2)
	
	loss = (1 - ((2.*intersection + smooth) / (pred.sum(dim=2).sum(dim=2)
			+ target.sum(dim=2).sum(dim=2) + smooth)))
	return loss.mean()
	
	
def plot_img_array(img_array, ncol=3):
	nrow = len(img_array) // ncol
	
	f, plots = plt.subplot(nrow, ncol, sharex = 'all', sharey = 'all',
							figsize=(ncol*4,nrow*4))
	for i in range(len(img_array))
		plots[i // ncol, i % ncol]
		plots[i // ncol, i % ncol].imshow(img_array[i])
		

from functools import reduce
def plot_side_by_side(img_arrays):
	flatten_list = reducce(lambda x,y: x+y, zip(*img_arrays))
	
	plot_img_array(np.array(flatten_list), ncol=len(img_arrays))
	
import itertools
def plot_errors(results_dict, title):
	markers = itertools.cycle(('+', 'x', 'o'))
	
	plt.title('{}'.format(title))
	
	for label, result in sorted(results_dict.items()):
		plt.plot(result, marker=next(markers), label=label)
		plt.ylabel('dice_coef')
		plt.xlabel('epoch')
		plt.legend(loc=3, bbox_to_anchor=(1,0))
		
	plt.show()
	
'''
def masks_to_colorimg(masks): ...
#채송: 어차피 우리 이미지가 x-ray다 보니 이 함수는 아예 참고 안했습니다. 사실 plot도 필요없긴 한데.. .. 혹시 모르니 일단은 추가해놓음
'''
