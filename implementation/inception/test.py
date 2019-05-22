import tensorflow as tf
from layers import layers
from tensorflow.keras.models import Model
from tensorflow.keras.losses import MSE
from tensorflow.keras.optimizers import Adam
from inception import Inception
import numpy as np

image = tf.random.normal(
		shape=[1,28,28,2], mean=0.0, stddev=1.0, dtype = tf.dtypes.float32)

def network(image):
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
	aux_net1_avgpool = layers.avgPool(inputs=inception4a, kernel_size=(5,5), stride=(3,3),padding="same")
	aux_net1_conv = layers.Conv2d(inputs=aux_net1_avgpool, filters=128, kernel_size=(1,1), strides=(2,2),name="aux1_conv")
	aux_net1_flatten = layers.Flatten(inputs=aux_net1_conv)
	aux_net1_dense1 = layers.Dense(units=1024, activation="relu", inputs=aux_net1_flatten,name="aux_net1_dense")
	aux_net1_dense2  = layers.Dense(units=1024, activation="softmax", inputs=aux_net1_dense1,name="aux_net1_dense2")

	inception4b = Inception.inception_module(inputs=inception4a,one_filters=160,three_filters=224,five_filters=64,three_filter_reduce=112,five_filter_reduce=24,pool_proj_filters=64,name="inception4b")
	inception4c = Inception.inception_module(inputs=inception4b,one_filters=128,three_filters=256,five_filters=64,three_filter_reduce=128,five_filter_reduce=64,pool_proj_filters=64,name="inception4c")
	inception4d = Inception.inception_module(inputs=inception4c,one_filters=112,three_filters=288,five_filters=64,three_filter_reduce=144,five_filter_reduce=32,pool_proj_filters=64,name="inception4d")
	# aux net 
	aux_net2_avgpool = layers.avgPool(inputs=inception4d, kernel_size=(5,5), stride=(3,3),padding="same")
	aux_net2_conv = layers.Conv2d(inputs=aux_net2_avgpool, filters=128, kernel_size=(1,1), strides=(2,2),name="aux1_conv")
	aux_net2_flatten = layers.Flatten(inputs=aux_net2_conv)
	aux_net2_dense1 = layers.Dense(units=1024, activation="relu", inputs=aux_net2_flatten,name="aux_net1_dense")
	aux_net2_dense2  = layers.Dense(units=1024, activation="softmax", inputs=aux_net2_dense1,name="aux_net1_dense2")


	inception4e = Inception.inception_module(inputs=inception4d,one_filters=256,three_filters=320,five_filters=128,three_filter_reduce=160,five_filter_reduce=32,pool_proj_filters=128,name="inception4e")
	max_pool2 = layers.maxPool(inputs=inception4e, kernel_size=(3,3), strides=(2,2),padding="same")
	inception5a =Inception.inception_module(inputs=max_pool2,one_filters=256,three_filters=320,five_filters=128,three_filter_reduce=160,five_filter_reduce=32,pool_proj_filters=128,name="inception5a")
	inception5b = Inception.inception_module(inputs=inception5a,one_filters=384,three_filters=384,five_filters=128,three_filter_reduce=192,five_filter_reduce=48,pool_proj_filters=128,name="inception5b")
	avg_pool = layers.avgPool(inputs=inception5b,kernel_size=(7,7),stride=(1,1),padding="same")
	#add drop out
	# flatten = layers.Flatten(inputs=avg_pool)
	dropout = layers.Dropout(inputs=avg_pool,rate=0.4)
	flatten = layers.Flatten(inputs=dropout)
	dense = layers.Dense(units=10,activation="softmax",inputs=flatten,name="dense")
	return dense


input_layer = tf.keras.layers.Input(shape=(28,28,1))
output_layer = network(input_layer)
training_model = Model(inputs=input_layer,outputs=output_layer)


mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train = (x_train.reshape(-1, 28, 28, 1) / 255).astype(np.float32)
# x_train, x_test = x_train / 255.0, x_test / 255.0
y_train = np.eye(10)[y_train].astype(np.float32)
x_train = x_train[:1]
y_train = y_train[:1]
print(y_train[0])
print(training_model)

for i in range(5):
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(training_model.trainable_variables)
        preds = training_model(x_train)
        loss = MSE(preds, y_train)
        grads = tape.gradient(loss, training_model.trainable_variables)
        optim.apply_gradients(zip(grads, training_model.trainable_variables))
print(loss)


