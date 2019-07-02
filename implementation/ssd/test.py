from vgg_16 import Vgg
from multishot import multishot
import tensorflow as tf
from configs import configs
from anchor_box import generate_anchors


vvg_out = Vgg()
num_classes = 1
input_layer = tf.keras.layers.Input(shape=(300,300,3))
modified_vgg = vvg_out(input_layer)
cfg = configs['number_of_boxes']
loc = multishot(modified_vgg, num_classes+1, cfg)
print(loc)
boxes = generate_anchors(configs)
print(len(boxes))