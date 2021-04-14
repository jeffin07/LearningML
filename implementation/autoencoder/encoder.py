import torch
import torch.nn as nn

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
