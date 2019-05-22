import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.layers import Conv2D,Dense,Flatten
from tensorflow.keras.models import Sequential
import numpy as np 


print(tf.__version__)
def simple_example():
	a = tf.constant([1,3],name="a")
	b = tf.constant([[2,1],[1,2]],name="b")
	multiply = tf.math.multiply(a,b,name="mul")
	with tf.Session() as sess:
		writer = tf.summary.FileWriter("directory path", sess.graph)
		print(sess.run(multiply))
	writer.close()

def tensorboard_tf2():
	network = Sequential([
		Conv2D(input_shape=[28,28,1], filters=64, kernel_size=3),
		Flatten(),
		Dense(units=32,activation="relu"),
		Dense(units=10,activation="softmax")
		])
# 	network = Sequential([
#     keras.layers.Flatten(input_shape=(28, 28)),
#     keras.layers.Dense(32, activation='relu'),
#     keras.layers.Dense(10, activation='softmax')
# ])


	network.compile(optimizer='adam',
	loss='sparse_categorical_crossentropy',
	metrics=['accuracy'])
	print(network)

	'''
	Import dataset
	'''
	mnist = tf.keras.datasets.mnist

	(x_train, y_train),(x_test, y_test) = mnist.load_data()
	x_train = (x_train.reshape(-1, 28, 28, 1) / 255).astype(np.float32)
	# x_train, x_test = x_train / 255.0, x_test / 255.0
	# y_train = np.eye(10)[y_train].astype(np.float32)
	# x_train1 = x_train[:20]
	# y_train1 = y_train[:20]
	logdir="logs/fit/"
	tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)


	network.fit(x_train,y_train,epochs=10,callbacks=[tensorboard_callback])
	# test = x_train[21:22]
	# print(tf.argmax(network.predict(test)))
	# print(y_train[21])

tensorboard_tf2()