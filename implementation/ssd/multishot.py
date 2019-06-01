from tensorflow.keras.layers import Conv2D


def multishot(features_layers, num_classes, cfg):
	'''

	Args

	extra_features : extra feature layers for SSD
	num_classes  : number of classes 
	cfg : conatains the number of boxes for each layer
	
	Output

	list of localization and classification

	'''

	loc_layers = []
	conf_layers = []
	for index, layer in enumerate(features_layers):
		loc_layers += [
			Conv2D(
				filters=cfg[index] * 4, kernel_size=3,
				padding="same")(layer)]
		conf_layers += [
			Conv2D(
				filters=cfg[index] * num_classes, kernel_size=3,
				padding="same")(layer)]
	return [loc_layers, conf_layers]
