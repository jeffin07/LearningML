import tensorflow as tf 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D,Dense,Flatten,MaxPool2D


class Vgg(Model):
	def __init__(self,output_nodes):
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
		self.dense1_1 = Dense(
					units=4096, activation="relu")
		self.dense1_2 = Dense(
					units=4096, activation="relu")
		self.dense2 = Dense(
					units=output_nodes, activation="softmax")
		self.maxPool = MaxPool2D(
					pool_size=2, strides=2, padding="same")
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
		x = self.maxPool(x)
		x = self.flatten(x)
		x = self.dense1_1(x)
		x = self.dense1_2(x)
		x = self.dense2(x)
		return out4_3, x