import torch.nn as nn



class ViT(nn.Module):

	def __init__(
			self, units=6, heads=8, input_dim=(512, 512),
			patch_size=16
		):

			super(ViT, self).__init__()


			# embed_path
			pass
