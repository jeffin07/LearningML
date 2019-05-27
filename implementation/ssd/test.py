from vgg_16 import Vgg
import tensorflow as tf

vvg_out = Vgg()
input_layer = tf.keras.layers.Input(shape=(300,300,3))
modified_vgg = vvg_out(input_layer)
for i in modified_vgg:
	print(i)