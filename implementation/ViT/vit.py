import torch
import torch.nn as nn



class Patch_embedding(nn.Module):

	"""
	
	input
	=====

	image_dim(int): dimension of the input image
	patch_dim(int): dimension of the pathch
	channels(int):  input image channels
	embed_size(int): size of the embedding 
	
	output
	======

	returns the input images split according to the patch size(tensor)

	"""

	def __init__(self, image_dim, patch_dim=16, channels=3, embed_size=768):

		super(Patch_embedding, self).__init__()

		self.image_dim = image_dim
		self.patch_dim = patch_dim
		self.embed_size = embed_size
		self.channels = channels
		# self.num_patches = (self.image_dim // self. patch_dim) ** 2

		self.conv = nn.Conv2d(in_channels=self.channels, out_channels = self.embed_size, kernel_size=self.patch_dim, stride=self.patch_dim)

	def forward(self, x):

		# proj.shape =  [batches, out_channels, kernel_size**2, kernel_size**2]
		proj = self.conv(x) 
		# we flatten on the second dimension ie the out_channels
		# out.shape = [batches, out_channels, *]
		out = torch.flatten(proj, start_dim=2)
		# we need to input to be of embed_size,[batches, * , out_channels]
		out = out.transpose(1, 2)
		return out


class MLP(nn.Module):

	def __init__(self, in_features, hidden_features, out_features, p=0.5):

		super(MLP, self).__init__()

		# linera->act->dropout->linear->dropout

		self.fc1 = nn.Linear(in_features, hidden_features)
		self.activation = nn.GELU()
		self.dropout = nn.Dropout(p)
		self.fc2 = nn.Linear(hidden_features, in_features)


	def forward(self, x):


		fc1 = self.activation(self.fc1(x))
		fc1 =  self.dropout(fc1)
		fc2 = self.fc2(fc1)
		out = self.dropout(fc2)

		return out


# class ViT(nn.Module):

# 	def __init__(
# 			self, units=6, heads=8, input_dim=(512, 512),
# 			patch_size=16
# 		):

# 			super(ViT, self).__init__()


# 			# embed_path
# 			pass


if __name__ == '__main__':


	input_tensor = torch.randn([1,3,96,96])


	patch_proj = Patch_embedding(96)

	result = patch_proj(input_tensor)

	print(result.shape)