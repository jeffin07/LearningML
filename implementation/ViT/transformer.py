import torch
import torch.nn as nn

from self_attention import multi_head_self_attention as mhsa

class TransformerBlock(nn.Module):

	def __init__(self, dim, heads=8, dropout=0.1):
		
		super(TransformerBlock, self).__init__()

		# multi head self attention
		# add an norm
		# linear
		# add and norm
		self.dropout = nn.Dropout(dropout)
		self.norm = nn.LayerNorm(dim)
		self.dim_linear = dim * 4
		self.linear = nn.Sequential(

				nn.Linear(dim, self.dim_linear),
				nn.ReLU(),
				self.dropout,
				nn.Linear(self.dim_linear, dim),
				self.dropout

		)
		self.multi_head_attention = mhsa(embed_size=dim, heads=heads)

	def forward(self, x):

		first_attention = self.multi_head_attention(x)
		out_1 = self.norm(first_attention + x )
		out =  self.norm(self.linear(out_1) + out_1)

		return out 


if __name__ == '__main__':

    x = torch.randn([1,4,512])

    tb = TransformerBlock(512)

    out = tb(x)

    print(out.shape)