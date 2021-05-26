import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils import data
import torch.optim as optim


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
		avg_pool = nn.AvgPool2d(7)
		num_classes = 10
		# input shape [batch, 3, 224, 224]
		layers.append(convblock(3, input_channel, stride=2))


		for t,c,n,s in self.conf:

			for num in range(n):
				stride = s if num ==0 else 1
				layers.append(invertedresidualblock(input_channel, t, c, stride=stride))
				input_channel = c
		

		layers.append(convblock(input_channel, last_channel, stride=1))
		layers.append(avg_pool)
		# layers.append(convblock(last_channel, 1000, stride=1))

		self.features = nn.Sequential(*layers)

		self.classifier = nn.Sequential(
			nn.Dropout(0.2),
			nn.Linear(last_channel,num_classes)
			)

	def __str__(self):

		return '{}'.format(self.features)

	def forward(self, x):
		x = self.features(x)
		x = x.reshape(x.shape[0], -1)
		return self.classifier(x)



if __name__ == '__main__':

	model = MobilenetV2()

	batch_size = 32

	dataset = datasets.CIFAR10(
		root='./data'
		,train=True
		,download=True
		,transform=transforms.Compose([
			transforms.CenterCrop(224),
			transforms.ToTensor()
		])
	)

	dataloader = data.DataLoader(
		dataset,
		shuffle=True,
		drop_last=True,
		batch_size=batch_size
	)

	optimizer = optim.Adam(params=model.parameters(), lr=0.01)
	criterion = nn.CrossEntropyLoss()

	for epoch in range(5):
		for batch_idx, (img, target) in enumerate(dataloader):
			# img = torch.reshape(img, [1,3,224,224])
			optimizer.zero_grad()
			output = model(img)
			loss = criterion(output, target)
			loss.backward() 
			optimizer.step()
			if batch_idx % 10 == 0:
				print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(img), len(dataloader.dataset),
				100. * batch_idx / len(dataloader), loss.item()))