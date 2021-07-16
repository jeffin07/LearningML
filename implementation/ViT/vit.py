import torch
import torch.nn as nn
from self_attention import multi_head_self_attention as attention


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
	"""

	parameters
	==========
	in_features (int) : size of the input dimension
	hidden_features (int) : hidden dimension
	out_features (int) : number of output features

	"""
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

class Block(nn.Module):


	"""
	
	input
	=====

	heads(int) : number of attention heads
	embed_size(int): input embeddig dimension


	"""

	def __init__(self, heads, embed_size=768, mlp_ratio=4.0):

		super(Block, self).__init__()

		self.norm1 = nn.LayerNorm(embed_size, 1e-6)
		self.norm2 = nn.LayerNorm(embed_size, 1e-6)
		self._attention = attention(heads=heads, embed_size=embed_size)

		hidden_features = int(embed_size * mlp_ratio)
		self._mlp = MLP(embed_size, hidden_features, embed_size)

	def forward(self, x):

		# norm1 = self.norm1(x)
		# att = self._attention(x)
		# print(att.shape, x.shape)
		out1 = x + self._attention(self.norm1(x))
		out2 = out1 + self._mlp(self.norm2(out1))

		return out2

		# transformer block
		#norm - > attention+skip -> norm -> mlp
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
	block = Block(heads=1)

	block(result)