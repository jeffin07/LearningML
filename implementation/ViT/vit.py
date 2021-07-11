import torch
import torch.nn as nn



class Patch_embedding(nn.Module):

	def __init__(self, image_dim, patch_dim=16, channels=3, embed_size=768):

		super(Patch_embedding, self).__init__()

		self.image_dim = image_dim
		self.patch_dim = patch_dim
		self.embed_size = embed_size
		self.channels = channels
		# self.num_patches = (self.image_dim // self. patch_dim) ** 2

		self.conv = nn.Conv2d(in_channels=self.channels, out_channels = self.embed_size, kernel_size=self.patch_dim, stride=self.patch_dim)

	def forward(self, x):

		proj = self.conv(x)
		out = torch.flatten(proj, start_dim=2)
		out = out.transpose(1, 2)
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