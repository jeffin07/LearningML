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
inception3a = Inception.inception_module(inputs=max_pool1,one_filters=64,three_filters=128,five_filters=32,three_filter_reduce=96,five_filter_reduce=16,pool_proj_filters=32,name="inception3a")
inception3b = Inception.inception_module(inputs=inception3a,one_filters=128,three_filters=192,five_filters=96,three_filter_reduce=128,five_filter_reduce=32,pool_proj_filters=64,name="inception3b")
max_pool2 = layers.maxPool(inputs=inception3b, kernel_size=(3,3), strides=(2,2),padding="same")
inception4a = Inception.inception_module(inputs=max_pool2,one_filters=192,three_filters=208,five_filters=48,three_filter_reduce=96,five_filter_reduce=16,pool_proj_filters=64,name="inception4a")
# aux net 
inception4b = Inception.inception_module(inputs=inception4a,one_filters=160,three_filters=224,five_filters=64,three_filter_reduce=112,five_filter_reduce=24,pool_proj_filters=64,name="inception4b")
inception4c = Inception.inception_module(inputs=inception4b,one_filters=128,three_filters=256,five_filters=64,three_filter_reduce=128,five_filter_reduce=64,pool_proj_filters=64,name="inception4c")
inception4d = Inception.inception_module(inputs=inception4c,one_filters=112,three_filters=288,five_filters=64,three_filter_reduce=144,five_filter_reduce=32,pool_proj_filters=64,name="inception4d")
# aux net 
inception4e = Inception.inception_module(inputs=inception4d,one_filters=256,three_filters=320,five_filters=128,three_filter_reduce=160,five_filter_reduce=32,pool_proj_filters=128,name="inception4e")
max_pool2 = layers.maxPool(inputs=inception4e, kernel_size=(3,3), strides=(2,2),padding="same")
inception5a =Inception.inception_module(inputs=max_pool2,one_filters=256,three_filters=320,five_filters=128,three_filter_reduce=160,five_filter_reduce=32,pool_proj_filters=128,name="inception5a")
inception5b = Inception.inception_module(inputs=inception5a,one_filters=384,three_filters=384,five_filters=128,three_filter_reduce=192,five_filter_reduce=48,pool_proj_filters=128,name="inception5b")
avg_pool = layers.avgPool(inputs=inception5b,kernel_size=(7,7),stride=(1,1),padding="same")
#add drop out
# flatten = layers.Flatten(inputs=avg_pool)
dropout = layers.Dropout(inputs=avg_pool,rate=0.4)
flatten = layers.Flatten(inputs=dropout)
dense = layers.Dense(units=1000,activation="relu",inputs=flatten,name="dense")
print(dense.shape)

