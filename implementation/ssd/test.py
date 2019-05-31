from vgg_16 import Vgg
from multishot import multishot
import tensorflow as tf

vvg_out = Vgg()
num_classes = 1
input_layer = tf.keras.layers.Input(shape=(300,300,3))
modified_vgg = vvg_out(input_layer)
loc = multishot(modified_vgg,num_classes+1)
print(loc)