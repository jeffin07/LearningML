from tensorflow.keras.layers import Conv2D

# not a good way should include config :/
mbox = [4, 6, 6, 6, 4, 4]


def multishot(features_layers, num_classes):
	'''

	Args

	extra_features : extra feature layers for SSD
	num_boxes  : number of boxes at each feature layer

	Output

	list of localization and classification

	'''

	loc_layers = []
	conf_layers = []
	for index, layer in enumerate(features_layers):
		loc_layers += [
			Conv2D(
				filters=mbox[index] * 4, kernel_size=3,
				padding="same")(layer)]
		conf_layers += [
			Conv2D(
				filters=mbox[index] * num_classes, kernel_size=3,
				padding="same")(layer)]
	return [loc_layers, conf_layers]
