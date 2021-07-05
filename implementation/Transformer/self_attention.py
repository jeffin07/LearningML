import torch
import torch.nn as nn



class multi_head_self_attention(nn.Module):
    '''

    A simple implemetation of self attention from the
    paper Attention is all you need https://arxiv.org/abs/1706.03762v5

    embed_size : embedding size of input sequence (512 in paper)
    heads :  number of heads
    mask :  for masked attention
    
    '''
    def __init__(self, embed_size, heads=1, mask=None):

        super(multi_head_self_attention, self).__init__()

        self.embed_size = embed_size
        self.heads = heads
        self.mask = mask
        self.head_dim = self.embed_size // self.heads
        self.scaled_factor = self.head_dim ** -0.5

        self.keys = nn.Linear(self.embed_size, self.head_dim, bias=False)
        self.values = nn.Linear(self.embed_size, self.head_dim, bias=False)
        self.query = nn.Linear(self.embed_size, self.head_dim, bias=False)

        self.fc_out = nn.Linear(self.head_dim, self.embed_size, bias=False)

        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):

        attention_heads = []

        for _ in range(self.heads):
            keys = self.keys(x)
            values = self.values(x)
            querys = self.query(x)

        # attention = torch.matmul(querys, torch.transpose(keys, 0, 1)) * self.scaled_factor
            attention_heads.append(self._attention(keys, querys, values, mask=self.mask))

        attention_concat  = torch.cat(attention_heads, dim=2)
        # out = self.fc_out()
        print("*****",len(attention_heads), attention_heads[0].shape)

        return attention_concat

    def _attention(self, keys, querys, values, mask=None):

        attention = torch.einsum("bij,bkj -> bik", querys, keys) #* self.scaled_factor
        if mask:
            mask = torch.ones(attention.shape)
            mask = torch.triu(mask,diagonal=1) * (-1e9)
            print("attention : ",attention)
            print("mask : ", mask)
            attention += mask
        print("again attention:",attention)
        attention = self.softmax(attention)
        print("again attention:",attention)
        out = torch.matmul(attention,values)
        return out

if __name__ == '__main__':

    x = torch.randn([1,4,512])

    mhsa = multi_head_self_attention(512, heads=2)

    mhsa(x)


