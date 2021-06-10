import torch
import torch.nn as nn


class self_attention(nn.Module):

    def __init__(self, embed_size, heads=1):

        super(self_attention, self).__init__()

        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = self.embed_size // self.heads

        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)

        self.query = nn.Linear(self.head_dim, self.head_dim, bias=False)

        # scaled dot product
        # q*v -> k 

    def forward(self, x):

        keys = self.keys(x)
        values = self.values(x)
        querys = self.query(x)
        soft = nn.Softmax(dim=1)
        out = torch.matmul(soft(torch.matmul(querys,keys)),values)
        return out

if __name__ == '__main__':

    x = torch.randn([3,3])

    sa = self_attention(3)

    print(sa(x))


