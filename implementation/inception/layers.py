import tensorflow as tf 


class layers:

	def Conv2d(inputs, filters, kernel_size, strides, name):
		return tf.keras.layers.Conv2D(
				filters=filters, kernel_size=kernel_size,
				strides=strides, name=name,
				activation="relu", padding="same")(inputs)

	def maxPool(inputs, kernel_size, strides, padding):
		return tf.keras.layers.MaxPool2D(
				pool_size=kernel_size, strides=strides,
				padding=padding)(inputs)

	def avgPool(inputs, kernel_size, stride, padding):
		return tf.keras.layers.AveragePooling2D(
				pool_size=kernel_size, strides=stride,
				padding=padding)(inputs)

	def concat(inputs, axis, name):
		return tf.keras.layers.concatenate(
				inputs=inputs, axis=axis, name=name)

	def Dense(inputs,units,activation,name):
		return tf.keras.layers.Dense(
				units=units,activation=activation,name=name)(inputs)

	def Flatten(inputs):
		return tf.keras.layers.Flatten()(inputs)

	def Dropout(inputs,rate):
		return tf.keras.layers.Dropout(rate=rate)(inputs)