import torch
import torch.nn as nn


# class convb 

class convblock(nn.Module):

	def __init__(
			self, in_channels, out_channels, kernel_size=1,
			stride=1, groups=1, dilation=1
	):

		super(convblock, self).__init__()
		padding = (kernel_size - 1) // 2 * dilation

		self.conv_layer = nn.Conv2d(
				in_channels=in_channels, out_channels=out_channels,
				kernel_size=kernel_size, padding=padding, stride=stride,
				groups=groups
			)
		self.batchnorm = nn.BatchNorm2d(out_channels)
		self.activation = nn.ReLU6(inplace=True)

	def forward(self, x):

		x = self.conv_layer(x)
		x = self.batchnorm(self.activation(x))
		return x

# class inverted-residual block

class invertedresidualblock(nn.Module):

	def __init__(
			self, input_channels ,expansion_ratio, out_channels, stride
	):
		super(invertedresidualblock, self).__init__()

		hidden_dim = int(expansion_ratio * input_channels)
		self.conv1 = nn.Sequential(
				# nn.Conv2d(input_channels, out_channels,1)
				convblock(input_channels, hidden_dim, stride),
				convblock(hidden_dim, hidden_dim, kernel_size=3 , stride=stride, groups=hidden_dim),
				convblock(hidden_dim, out_channels)
			)

	def forward(self, x):

		return(self.conv1(x))




# class Mobilenetv2
class MobilenetV2(nn.Module):

	def __init__(self):

		super(MobilenetV2, self).__init__()
		self.conf = [
			#t,c,n,s
			[1,16,1,1],
			[6,24,2,2],
			[6,32,3,2],
			[6,64,4,2],
			[6,96,3,1],
			[6,160,3,2],
			[6,320,1,1],
		]

		layers = []
		input_channel = 32
		last_channel = 1280
		avg_pool = nn.AdaptiveAvgPool2d(1)
		# def forward(self, x):
		layers.append(convblock(3, input_channel, stride=2))
		# for i in self.conf:

		for t,c,n,s in self.conf:

			for num in range(n):
				layers.append(invertedresidualblock(input_channel, c, t, stride=s))
				input_channel = c


		layers.append(convblock(input_channel, last_channel, stride=1))
		layers.append(avg_pool)
		layers.append(convblock(input_channel, 1000, stride=1))

		self.features = nn.Sequential(*layers)

	def __str__(self):

		return '{}'.format(self.features)

	def forward(self, x):

		return self.features(x)



if __name__ == '__main__':


	input_tensor = torch.rand(1,3,224,224)

	inv = invertedresidualblock(32, 1, 16, stride=1)
	conv = convblock(3, 32, stride=2)
	# Error in forward function
	# test convb and inverted seperately
	model = MobilenetV2()
	print(model(input_tensor))
	# print(inv(conv(input_tensor)).size())