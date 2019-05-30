import tensorflow as tf 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D,Dense,Flatten,MaxPool2D


class Vgg(Model):
	'''

	VGG implementation for SSD

	changed the original VGG strucure by changing fc6,fc7 into conv layers
	a maxpool after conv4_3 and added extra feature layers by removing last
	fc layer

	Input

	An tensor(image) with size [batch, 300, 300, 3]

	Output

	a list of feature layers for SSD to perform localzation and classification

	'''
	def __init__(self):
		super(Vgg, self).__init__()
		# layers needed
		self.conv1_1 = Conv2D(
					input_shape=[None,300,300,3],filters=64,
					kernel_size=3, padding="same", activation="relu")
		self.conv1_2 = Conv2D(
					filters=64, kernel_size=3,
					padding="same", activation="relu")
		self.conv2_1 = Conv2D(
					filters=128, kernel_size=3,
					padding="same", activation="relu")
		self.conv2_2 = Conv2D(
					filters=128, kernel_size=3,
					padding="same", activation="relu")
		self.conv3_1 = Conv2D(
					filters=256,kernel_size=3,
					padding="same", activation="relu")
		self.conv3_2 = Conv2D(
					filters=256,kernel_size=3,
					padding="same", activation="relu")
		self.conv3_3 = Conv2D(
					filters=256,kernel_size=3,
					padding="same", activation="relu")
		
		self.conv4_1 = Conv2D(
					filters=512, kernel_size=3,
					padding="same", activation="relu")
		self.conv4_2 = Conv2D(
					filters=512, kernel_size=3,
					padding="same", activation="relu")
		self.conv4_3 = Conv2D(
					filters=512, kernel_size=3,
					padding="same", activation="relu")
		self.conv5_1 = Conv2D(
					filters=512, kernel_size=3,
					padding="same", activation="relu")
		self.conv5_2 = Conv2D(
					filters=512, kernel_size=3,
					padding="same", activation="relu")
		self.conv5_3 = Conv2D(
					filters=512, kernel_size=3,
					padding="same", activation="relu")
		# This must be atrous_conv2d
		# self.conv6 = tf.nn.atrous_conv2d(
		# 	value=tf.constant([-1,19,19,512]),
		# 	filters=tf.constant([1,1,512,512]), rate=2, padding="same")
		self.conv6 = Conv2D(
					filters=1024, kernel_size=1,
					padding="same", activation="relu", dilation_rate=2)
		self.conv7 = Conv2D(
					filters=1024, kernel_size=1,
					padding="same", activation="relu", dilation_rate=2)
		self.conv8_1 = Conv2D(
					filters=256, kernel_size=1,
					padding="same", activation="relu")
		self.conv8_2 = Conv2D(
					filters=512, kernel_size=3,
					padding="same", strides=2, activation="relu")
		self.conv9_1 = Conv2D(
					filters=128, kernel_size=1,
					padding="same", activation="relu")
		self.conv9_2 = Conv2D(
					filters=256, kernel_size=3,
					padding="same", strides=2, activation="relu")
		self.conv10_1 = Conv2D(
					filters=128, kernel_size=1,
					padding="same", activation="relu")
		self.conv10_2 = Conv2D(
					filters=256, kernel_size=3,
					padding="valid",  activation="relu")
		self.conv11_1 = Conv2D(
					filters=128, kernel_size=1,
					padding="same", activation="relu")
		self.conv11_2 = Conv2D(
					filters=256, kernel_size=3,
					padding="valid",  activation="relu")

		'''
		self.dense1_1 = Dense(
					units=4096, activation="relu")
		self.dense1_2 = Dense(
					units=4096, activation="relu")
		self.dense2 = Dense(
					units=output_nodes, activation="softmax")
		'''
		self.maxPool = MaxPool2D(
					pool_size=2, strides=2, padding="same")
		self.maxPool5 = MaxPool2D(
					pool_size=2, strides=1, padding="same")
		self.flatten = Flatten()

	def call(self,input):
		# ops 
		x = self.conv1_1(input)
		x = self.conv1_2(x)
		x = self.maxPool(x)
		x = self.conv2_1(x)
		x = self.conv2_2(x)
		x = self.maxPool(x)
		x = self.conv3_1(x)
		x = self.conv3_2(x)
		x = self.conv3_3(x)
		x = self.maxPool(x)
		x = self.conv4_1(x)
		x = self.conv4_2(x)
		out4_3 = self.conv4_3(x)
		x = self.maxPool(out4_3)
		x = self.conv5_1(x)
		x = self.conv5_2(x)
		x = self.conv5_3(x)
		x = self.maxPool5(x)
		fc6 = self.conv6(x)
		fc7 = self.conv7(fc6)
		x = self.conv8_1(fc7)
		conv8_2 = self.conv8_2(x)
		x = self.conv9_1(conv8_2)
		conv9_2 = self.conv9_2(x) 
		x = self.conv10_1(conv9_2)
		conv10_2 = self.conv10_2(x)
		x = self.conv11_1(conv10_2)
		conv11_2 = self.conv11_2(x)

		'''
		original Vgg arch
		x = self.flatten(x)
		x = self.dense1_1(x)
		x = self.dense1_2(x)
		x = self.dense2(x)
		'''

		# return [fc6] 
		return [out4_3, fc7, conv8_2, conv9_2, conv10_2, conv11_2]