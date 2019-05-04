from layers import layers


class Inception:

	def inception_layer(
			inputs, one_filters, three_filters, three_filter_reduce,
			five_filters, five_filter_reduce, pool_proj_filters, name):
		one_one = layers.Conv2d(
				inputs=inputs, filters=one_filters, kernel_size=(1,1),
				strides=(1,1), name=name+"1_1")
		three_reduce = layers.Conv2d(
				inputs=inputs, filters=three_filter_reduce,
				kernel_size=(1,1), strides=(1,1), name=name+"3reduce")
		three_three = layers.Conv2d(
				inputs=three_reduce, filters=three_filters, kernel_size=(3,3),
				strides=(1,1), name=name+"3_3")
		five_reduce = layers.Conv2d(
				inputs=inputs, filters=five_filter_reduce,
				kernel_size=(1,1), strides=(1,1), name=name+"5reduce")
		five_five = layers.Conv2d(
				inputs=five_reduce, filters=five_filters, kernel_size=(5,5),
				strides=(1,1), name=name+"5_5")
		max_pool = layers.maxPool(
				inputs=inputs, kernel_size=(3,3), strides=(1,1),
				padding="same")
		pool_proj = layers.Conv2d(inputs=max_pool,
				filters=pool_proj_filters, kernel_size=(1,1),
				strides=(1,1), name=name+"pool_proj")
		return layers.concat(
				inputs=[one_one, three_three, five_five, pool_proj],
				axis=3, name=name+"concat")