import torch
import torch.nn.functional as F
import torch.optim as optim
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data
import torchvision.datasets.mnist as mnist
import torchvision
from model import Net
'''
root = "/Users/linhan/Desktop/cv/mnist/data/MNIST/raw"
train_set = (
    mnist.read_image_file(os.path.join(root, 'train-images-idx3-ubyte')),
    mnist.read_label_file(os.path.join(root, 'train-labels-idx1-ubyte'))
        )
train_dataset = torch.utils.data.TensorDataset(train_set[0], train_set[1])
train_loader = torch.utils.data.DataLoader(train_dataset,
                                 batch_size=8,
                                 shuffle=True,
                                 drop_last=True)
'''
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=8, shuffle=True)
def train(net, train_loader, optimizer):
    #net.train()
    for batch_id, (data,target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = net(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_id%750 == 0:
            print(f"loss = {loss.item()}      [{batch_id*len(data)}/{len(train_loader.dataset)}]")


net = Net()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.5)
epoch = 0
while epoch < 20:
    epoch = epoch + 1
    print(f"epoch: {epoch}")
    train(net, train_loader, optimizer)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=8, shuffle=True)

test_loss = 0
correct = 0
for data, target in test_loader:
    output = net(data)
    test_loss += F.nll_loss(output, target).item()
    pred = output.data.max(1, keepdim=True)[1]
    correct += pred.eq(target.data.view_as(pred)).sum()
test_loss /= len(test_loader.dataset)
print(f"\nTest loss = {test_loss}, Accuracy = {100.*correct/len(test_loader.dataset)}")