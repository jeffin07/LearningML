import torch
import torch.nn as nn


class multi_head_self_attention(nn.Module):

    def __init__(self, embed_size, heads=1):

        super(multi_head_self_attention, self).__init__()

        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = self.embed_size // self.heads

        self.keys = nn.Linear(self.embed_size, self.head_dim, bias=False)
        self.values = nn.Linear(self.embed_size, self.head_dim, bias=False)
        self.query = nn.Linear(self.embed_size, self.head_dim, bias=False)

        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):

        keys = self.keys(x)
        values = self.values(x)
        querys = self.query(x)

        attention = torch.matmul(querys, torch.transpose(keys, 0, 1))
        attention = self.softmax(attention)
        out = torch.matmul(attention,values)

        return out

if __name__ == '__main__':

    x = torch.randn([4,64])

    mhsa = multi_head_self_attention(64, heads=8)

    print(mhsa(x))


