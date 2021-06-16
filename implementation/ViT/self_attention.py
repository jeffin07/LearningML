import torch
import torch.nn as nn



# This is single head self attention
# implement another class for multi-head self attention
# [B,token,head]



class multi_head_self_attention(nn.Module):

    def __init__(self, embed_size, heads=1):

        super(multi_head_self_attention, self).__init__()

        self.embed_size = embed_size
        self.heads = heads
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
            attention_heads.append(self._attention(keys, querys, values))

        attention_concat  = torch.cat(attention_heads, dim=2)
        # out = self.fc_out()
        print("*****",len(attention_heads), attention_heads[0].shape)

        return attention_concat

    def _attention(self, keys, querys, values, mask=None):

        attention = torch.einsum("bij,bkj -> bik", querys, keys) #* self.scaled_factor
        attention = self.softmax(attention)
        out = torch.matmul(attention,values)

        return out

if __name__ == '__main__':

    x = torch.randn([1,4,512])

    mhsa = multi_head_self_attention(512, heads=8)

    print(mhsa(x).shape)


