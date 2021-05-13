import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils import data
import torch.optim as optim


class SELayer(nn.Module):

    def __init__(self, channels, reduction=16):

        super(SELayer, self).__init__()
        '''
            this has some more parameters and also initialize layers
            avgpool -> linear -> relu -> linear -> sigmoid
        '''
        self.avg_pool = nn.AdaptiveAvgPool2d(1)#nn.AvgPool2d(kernel_size=(1,1))
        # x_ = self.avg_pool(x)
        resized_out = channels // reduction
        # print("resized_out", resized_out)
        self.sqe = nn.Sequential(
            # nn.Flatten(),
            nn.Linear(channels, resized_out),
            nn.ReLU(),
            nn.Linear(resized_out, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        '''
            forward pass
        '''
        # print("x : ", x.size())
        batch,channels,_,_ = x.size()
        x_ = self.avg_pool(x).view(batch, channels)
        # print("x_ : ",x_.shape)
        # x_ = x_.view(h,w)
        out = self.sqe(x_).view(batch,channels,1,1)
        # print("out : ", out.shape)
        out = x * out.expand_as(x)
        # print("output : ", out.shape)
        return out



class backbone(nn.Module):

    def __init__(self):

        super(backbone, self).__init__()

        self.net = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3),
        nn.ReLU(),
        SELayer(16),
        nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
        nn.ReLU(),
        SELayer(32),
        nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features = 32, out_features = 10),
            nn.Softmax(dim=1)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):

        out =  self.avg_pool(self.net(x))
        # print(out.size())
        out = out.reshape(out.shape[0], -1)
        return self.fc(out)





if __name__ == '__main__':


    
    b_net = backbone()

    print(b_net)

    batch_size = 32

    dataset = datasets.MNIST(
        root='./data'
        ,train=True
        ,download=True
        ,transform=transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
    )

    dataloader = data.DataLoader(
        dataset,
        shuffle=True,
        drop_last=True,
        batch_size=batch_size
    )

    backbone = backbone()
    optimizer = optim.Adam(params=backbone.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(5):
        for batch_idx, (img, target) in enumerate(dataloader):

            optimizer.zero_grad()
            output = backbone(img)
            loss = criterion(output, target)
            loss.backward() 
            optimizer.step()
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(img), len(dataloader.dataset),
                    100. * batch_idx / len(dataloader), loss.item()))


