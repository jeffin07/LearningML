import torch
import torch.nn as nn

from torchvision import datasets, transforms
import torch.optim as optim
import matplotlib.pyplot as plt

class AutoEncoder(nn.Module):

	def __init__(self):

		super(AutoEncoder, self).__init__()

		# encoder
		self.encoder = nn.Sequential(
				nn.Linear(28*28, 64),
				nn.ReLU(),
				nn.Linear(64, 32),
				nn.ReLU(),
				nn.Linear(32, 16),
				nn.ReLU(),
				nn.Linear(16, 8),
		)

		# decoder

		self.decoder = nn.Sequential(
				nn.Linear(8, 16),
				nn.ReLU(),
				nn.Linear(16, 32),
				nn.ReLU(),
				nn.Linear(32, 64),
				nn.ReLU(),
				nn.Linear(64, 28*28),
				nn.Sigmoid()
		)



	def forward(self, x):

		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
		return decoded



if __name__ == '__main__':

	model = AutoEncoder() 
	print(model)

	transform = transforms.ToTensor()

	mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
	data_loader = torch.utils.data.DataLoader(dataset=mnist_data, batch_size=12,shuffle=True)

	criterion = nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

	num_epochs = 5
	output = []
	for epochs in range(num_epochs):
		for img,_ in data_loader:
			img = img.reshape(-1, 28*28)
			recon = model(img)
			loss = criterion(recon,img)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		print("epoch ",epochs,"loss ",loss)
		output.append((epochs,img,recon))

	for i in range(0, num_epochs, 4):
		plt.figure(figsize=(9,2))
		plt.gray()
		imgs = output[i][1].detach().numpy()
		recon = output[i][2].detach().numpy()
		for i,item in enumerate(imgs):
			if i >=9:
				break
			item = item.reshape(-1, 28, 28)
			plt.subplot(2,9,i+1)
			plt.imshow(item[0])
		for i,item in enumerate(recon):
			if i>=9:
				break
			print(item.shape)
			item=item.reshape(-1, 28, 28)
			print(item.shape)
			plt.subplot(2,9,9+i+1)
			plt.imshow(item[0])
	plt.show()
