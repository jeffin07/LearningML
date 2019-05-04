import tensorflow as tf
from layers import layers
from inception import Inception

image = tf.random.normal(
		shape=[1,224,224,3], mean=0.0, stddev=1.0, dtype = tf.dtypes.float32)

conv = layers.Conv2d(inputs=image, filters=64, kernel_size=(7,7), strides=(2,2), name="1")
max_pool = layers.maxPool(inputs=conv, kernel_size=(3,3), strides=(2,2),padding="same")
red=layers.Conv2d(inputs=max_pool, filters=64,kernel_size=(1,1), strides=(1,1), name="3reduce")
conv1 = layers.Conv2d(inputs=red, filters=192, kernel_size=(3,3), strides=(1,1), name="3")
max_pool1 = layers.maxPool(inputs=conv1, kernel_size=(3,3), strides=(2,2),padding="same")
inception1 = Inception.inception_layer(inputs=max_pool1,one_filters=64,three_filters=128,five_filters=32,three_filter_reduce=96,five_filter_reduce=16,pool_proj_filters=32,name="inception")
print(inception1.shape)

