import torch
import torch.nn as nn


class SELayer(nn.Module):

	def __init__(self, channels, reduction=16):

		super(SELayer, self).__init__()
		'''
			this has some more parameters and also initialize layers
			avgpool -> linear -> relu -> linear -> sigmoid
		'''
		self.avg_pool = nn.AvgPool2d(kernel_size=(1,1))
		# x_ = self.avg_pool(x)
		resized_out = channels // reduction
		self.sqe = nn.Sequential(
			nn.Linear(channels, resized_out),
			nn.ReLU(),
			nn.Linear(resized_out, channels),
			nn.Sigmoid()
		)

	def forward(self, x):
		'''
			forward pass
		'''
		h,w,c = x.shape()
		x_ = self.avg_pool(x)
		out = self.sqe(x_)
		out = x * out.expand_as(x)
		return out


if __name__ == '__main__':


	
	conv_1 = nn.Conv2d(out_channels=96, in_channels=1, kernel_size=11, stride=4)

	input_t =  torch.randn(20, 1, 244, 244)
	print(input_t.shape)

	res = conv_1(input_t)
	print(res.shape)
	SELayer(96)