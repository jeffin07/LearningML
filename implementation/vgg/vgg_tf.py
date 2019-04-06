import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
from keras.preprocessing import image
import data_pipeline

x = tf.placeholder(tf.float32 , [None, 224, 224, 3])
y = tf.placeholder(tf.float32 , [None, 2])
class vgg:

	def __init__(self, heigth ,width, channels, num_classes):

		self.heigth = heigth
		self.width = width
		self.channels = channels
		self.num_classes = num_classes
	def build(self):
		# x = tf.placeholder(tf.float32 , [None, 224, 224, 3])

		#first
		conv1_1 = tf.layers.conv2d(inputs=x, filters=64, kernel_size=3, activation=tf.nn.relu, padding='same', name='conv1')
		conv1_2 = tf.layers.conv2d(inputs=conv1_1, filters=64, kernel_size=3, activation=tf.nn.relu, padding='same', name='conv2')
		max_pool1_1 = tf.contrib.layers.max_pool2d(inputs=conv1_2, kernel_size=[2, 2], stride=[2, 2], padding='same')
		#second
		conv2_1 = tf.layers.conv2d(inputs=max_pool1_1, filters=128, kernel_size=3, activation=tf.nn.relu, padding='same', name='conv3')
		conv2_2 = tf.layers.conv2d(inputs=conv2_1, filters=128, kernel_size=3, activation=tf.nn.relu, padding='same', name='conv4')
		max_pool_2_1 = tf.contrib.layers.max_pool2d(inputs=conv2_2, kernel_size=[2, 2], stride=[2, 2], padding='same')
		#third
		conv3_1 = tf.layers.conv2d(inputs=max_pool_2_1, filters=256, kernel_size=3, activation=tf.nn.relu, padding='same', name='conv5')
		conv3_2 = tf.layers.conv2d(inputs=conv3_1, filters=256, kernel_size=3,  activation=tf.nn.relu, padding='same', name='conv6')
		conv3_3 = tf.layers.conv2d(inputs=conv3_2, filters=256, kernel_size=3, activation=tf.nn.relu, padding='same', name='conv7')
		max_pool_3_1 = tf.contrib.layers.max_pool2d(inputs=conv3_3, kernel_size=[2, 2], stride=[2, 2], padding='same')
		#fourth
		conv4_1 = tf.layers.conv2d(inputs=max_pool_3_1, filters=512, kernel_size=3, activation=tf.nn.relu, padding='same', name='conv8')
		conv4_2 = tf.layers.conv2d(inputs=conv4_1, filters=512, kernel_size=3, activation=tf.nn.relu, padding='same', name='conv9')
		conv4_3 = tf.layers.conv2d(inputs=conv4_2, filters=512, kernel_size=3, activation=tf.nn.relu, padding='same', name='conv10')
		max_pool_4_1 = tf.contrib.layers.max_pool2d(inputs=conv4_3, kernel_size=[2, 2], stride=[2, 2], padding='same')
		#fifth
		conv5_1 = tf.layers.conv2d(inputs=max_pool_4_1, filters=512, kernel_size=3, activation=tf.nn.relu, padding='same', name='conv11')
		conv5_2 = tf.layers.conv2d(inputs=conv5_1, filters=512, kernel_size=3, activation=tf.nn.relu, padding='same', name='conv12')
		conv5_3 = tf.layers.conv2d(inputs=conv5_2, filters=512, kernel_size=3, activation=tf.nn.relu, padding='same', name='conv13')
		max_pool_5_1 = tf.contrib.layers.max_pool2d(inputs=conv5_3, kernel_size=[2, 2], stride=[2, 2], padding='same')
		#flatten
		flatten1 = tf.contrib.layers.flatten(inputs=max_pool_5_1)
		#fc layers
		fc_1 = tf.layers.dense(inputs=flatten1, activation=tf.nn.relu, units=4096)
		fc_2 = tf.layers.dense(inputs=fc_1, activation=tf.nn.relu, units=4096)
		fc_3 = tf.layers.dense(inputs=fc_2, activation=None, units=self.num_classes)

		return fc_3
	



sample_data = data_pipeline.Dataset('path/to/dataset')
sample_data.read_dataset()
epochs = 100
net = vgg(224,224,3,2)
k = net.build()
hello = tf.constant("hello")
loss = (tf.nn.softmax_cross_entropy_with_logits(logits=k ,labels=y))
global_step = tf.Variable(0, name='global_step', trainable=False)
cost = (tf.reduce_mean(loss))
optimizer = (tf.train.AdamOptimizer(0.1).minimize(cost))
c_pred = tf.equal(tf.argmax(k,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(c_pred,tf.float32))
with tf.Session() as sess :
	# print(sess.run(hello))
	sess.run(tf.global_variables_initializer())
	# tf.global_variables_initializer().run
	count =0
	for epoch in range(epochs):
		batches = sample_data.get_batch()
		for img, labels in batches:
			loss1, _,_ =(sess.run([accuracy,cost,optimizer],feed_dict={x:img,y:labels}))
		print("Accuracy at {} is {}".format(epoch, loss1))
