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
        self.avg_pool = nn.AvgPool2d(kernel_size=(1,1))
        # x_ = self.avg_pool(x)
        resized_out = channels // reduction
        self.sqe = nn.Sequential(
            nn.Linear(channels, resized_out),
            nn.ReLU(),
            nn.Linear(resized_out, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        '''
            forward pass
        '''
        h,w,c = x.shape()
        x_ = self.avg_pool(x)
        out = self.sqe(x_)
        out = x * out.expand_as(x)
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
        SELayer(16),
        nn.Flatten(),
        nn.Linear(in_features = 16 * 32 * 32, out_features = 10),
        nn.Softmax()

        )

    def forward(self, x):

        return self.net(x)





if __name__ == '__main__':


    
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
        batch_size=2
    )

    backbone = backbone()
    optimizer = optim.Adam(params=backbone.parameters(), lr=0.0001)

    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    for epoch in range(4):
        lr_scheduler.step()
        for imgs, classes in dataloader:
            # imgs, classes = imgs.to(device), classes.to(device)

            # calculate the loss
            output = backbone(imgs)
            loss = F.cross_entropy(output, classes)

            print(loss)
            # update the parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
