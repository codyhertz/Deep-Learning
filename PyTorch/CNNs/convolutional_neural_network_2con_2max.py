import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np


train_dataset = dsets.MNIST(root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True
                            )

test_dataset = dsets.MNIST(root='./data',
                           train=False,
                           transform=transforms.ToTensor()
                           )

batch_size = 100

n_iters = 3000

num_epochs = n_iters / (len(train_dataset) / batch_size)
num_epochs = int(num_epochs)

# create an iterable object of the training data
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True
                                           )

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=batch_size,
                                           shuffle=False
                                           )

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        # conv 1
        self.con1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()

        # max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        # conv 2
        self.con2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()

        # max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        # fully connected out
        self.fc1 = nn.Linear(32*7*7, 10)

    def forward(self, x):
        out = self.con1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)
        out = self.con2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)

        # resize, origianl size = 100 x 32 x 7 x 7
        # new 100 x 32*7*7
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out


model = CNNModel()
model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

i = 0
for epoch in range(num_epochs):
    for j, (images, labels) in enumerate(train_loader):
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        i += 1

        if i % 500 == 0:
            correct = 0
            total = 0

            for images, labels in test_loader:
                images = Variable(images.cuda())

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted.cpu() == labels.cpu()).sum()

            accuracy = 100.00 * float(correct) / float(total)

            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(i, loss.data, accuracy))
