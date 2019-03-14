import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
x = tf.placeholder(tf.float32 , [None, 224, 224, 3])
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
		fc_3 = tf.layers.dense(inputs=fc_2, activation=tf.nn.softmax, units=self.num_classes)

		return fc_3
	

net = vgg(224,224,3,2)
k = net.build()
print("Build Sucessfull")
print(net)
print(k)
img  = plt.imread('/home/jeffin/Downloads/test-224*224.jpeg')
print(img.shape)
img = img.reshape(1,224,224,3)
print(img.shape)

x1 = tf.ones([1,224,224,3], dtype=tf.float32)
# x = np.random.randn(1,224,224,3)
# x = tf.Variable(x, dtype=tf.float32)
# x = tf.Variable(x, dtype=tf.float32)
p = tf.ones([224,224,3], dtype=tf.float32)
init_op = tf.global_variables_initializer()
print(type(x1))
# print(p.shape)
# input()

with tf.Session() as sess:
	sess.run(init_op)
	# print(sess.run(x))
	# p = tf.Variable(tf.truncated_normal([1, 224, 224, 3], stddev=0.1))
	
	print(sess.run(k, feed_dict={x:img}))