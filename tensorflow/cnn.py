# -*- coding: utf-8 -*-
"""Untitled5.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1MmaMKY2MlckOCahjx8EMbTvHIE_idATy
"""

from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/MNIST', one_hot = True)

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# %matplotlib inline
tf.reset_default_graph()

y_check = data.train.labels[0:9, :]
x_check = data.train.images[0:9, :]
for i in range(9):
  img = x_check[i].reshape(28,28)
  print("Image of {} ".format(np.argmax(y_check[i])))
  plt.xticks([])
  plt.yticks([])
  plt.imshow(img)
  plt.show()

img_size = 28
img_shape = (img_size, img_size)
img_flat = img_size * img_size
num_classes = 10

def new_weights(shape):
  return tf.Variable(tf.truncated_normal(shape,stddev = 0.05))

def new_bias(shape):
  return tf.Variable(tf.constant(0.05,shape = [shape]))

conv1_filter_size = 5
conv1_number_filters = 16

conv2_filter_size = 7
conv2_number_filters = 36

fc_layer = 128

def new_conv(inputs, filter_size, filter_num, num_channels,pooling=True ):
  shape = [filter_size, filter_size, num_channels, filter_num]
  weight = new_weights(shape)
  bias = new_bias(filter_num)
  layer = tf.nn.conv2d(input = inputs, filter = weight, strides=[1,1,1,1], padding='SAME')
  print(layer.shape)
  print(bias.shape)
  layer += bias
  
  if pooling :
    layer = tf.nn.max_pool(value = layer, ksize = [1,2,2,1], strides = [1, 2, 2, 1], padding="SAME")
    
  layer = tf.nn.relu(layer)
  
  return layer, weight

def flatten_layer(layer):
  layer_shape = layer.get_shape()
  num_features = np.array(layer_shape[1:4], dtype=int).prod()
  
  layer_flatten = tf.reshape(layer,[-1, num_features])
  
  
  return layer_flatten, num_features

def new_fcl(layer, num_inputs, num_outputs, is_relu=True):
  
  weights = new_weights([num_inputs,num_outputs])
  bias = new_bias(num_outputs)
  
  fc_layer = tf.matmul(layer, weights) + bias
  
  if is_relu:
    fc_layer = tf.nn.relu(fc_layer)
  
  
  return fc_layer

x = tf.placeholder(tf.float32, [None, img_flat])
x_image = tf.reshape(x, [-1,img_size, img_size, 1])

y_true = tf.placeholder(tf.float32,[None, 10])
y_true_cls = tf.argmax(y_true)

layer_con1, layer_weight1 = new_conv(x_image, conv1_filter_size, conv1_number_filters, 1, pooling=True)

layer_con1

layer_con2 ,layer_weight2 = new_conv(layer_con1,  conv2_filter_size, conv2_number_filters, conv1_number_filters,  pooling=True)

layer_con2

layer_flat, num_features = flatten_layer(layer_con2)

layer_flat

num_features

fc_1 = new_fcl(layer_flat, num_features, fc_layer, is_relu=True)

fc_1

fc_2 = new_fcl(fc_1, fc_layer, num_classes, is_relu=False)

fc_2

y_pred = tf.nn.softmax(fc_2)
y_pred_cls = tf.argmax(y_pred)
y_pred_cls

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = fc_2,
                                                       labels = y_true)

cost = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate = 1e-4).minimize(cost)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session = tf.Session()
session.run(tf.initialize_all_variables())

batch_size =64
def optimize(num_iterations):
  for i in range(num_iterations):
    x_batch,y_true_batch = data.train.next_batch(batch_size)
    feed_dict_train = {x:x_batch,y_true:y_true_batch}
    
    session.run(optimizer, feed_dict = feed_dict_train)
    
    if i%100 == 0:
      acc = session.run(accuracy, feed_dict = feed_dict_train)
      loss = session.run(cost,feed_dict = feed_dict_train)
      print("Accuracy {} ".format(acc*100))
      print("loss at {} iteration is {}".format(i,loss))

optimize(202)

