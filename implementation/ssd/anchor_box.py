import itertools
import numpy as np
from math import sqrt


def generate_anchors(configs):
	boxes = []
	for indeces in range(len(configs['feature_maps'])):
		scale = configs['min_dim'] / configs['steps'][indeces]
		for j,i in itertools.product(range(configs['feature_maps'][indeces]), repeat=2):
			x_center = (i + 0.5) / scale
			y_center = (j + 0.5) / scale
			print(x_center,y_center)

			# small box
			size = configs['min_sizes'][indeces]
			h = w = size / configs['min_dim']
			boxes.append([x_center, y_center, h, w])


			# large box
			size = np.sqrt(configs['min_sizes'][indeces] * configs['max_sizes'][indeces])
			h = w = size / configs['min_dim']
			boxes.append([x_center, y_center, h, w])

			# aspect ratios
			size = configs['min_sizes'][indeces]
			h = w = size / configs['min_dim']
			for ratios in (configs['aspect_ratios'][indeces]):
				ratio = sqrt(ratios)
				boxes.append([x_center,y_center,h*ratio,w/ratio])
				boxes.append([x_center,y_center,h/ratio,w*ratio])
	boxes = np.array(boxes)
	if configs['clip']:
		boxes = np.clip(boxes, 0.0, 1.0)
	return boxes