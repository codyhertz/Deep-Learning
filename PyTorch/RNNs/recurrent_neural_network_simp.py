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


class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, layer_dim, output_size):
        super(RNNModel, self).__init__()

        # number of nodes per layer
        self.hidden_size = hidden_size

        # number of hidden layers
        self.layer_dim = layer_dim

        # building rnn
        # batch_first=True causes input/output tensors to be (batch_dim, seq_dim, input_dim)
        self.rnn = nn.RNN(input_size, hidden_size, layer_dim, batch_first=True, nonlinearity='relu')

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # initialize hidden state with zeros
        # (layer_dim, batch_size, hidden_size)
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_size).cuda())

        # one time step
        out, hn = self.rnn(x, h0)
        # index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step of hidden states
        out = self.fc(out[:, -1, :])
        return out


input_size = 28
hidden_size = 100
layer_dim = 1
output_size = 10

model = RNNModel(input_size, hidden_size, layer_dim, output_size)
model.cuda()

criterion = nn.CrossEntropyLoss()

learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

sequence_size = 28

i = 0
for epoch in range(num_epochs):
    for j, (images, labels) in enumerate(train_loader):
        images = Variable(images.view(-1, sequence_size, input_size).cuda())
        labels = Variable(labels.cuda())

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward() # get new gradients
        optimizer.step() # updating params

        i += 1

        if i % 500 == 0:
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = Variable(images.view(-1, sequence_size, input_size).cuda())

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)

                correct += (predicted.cpu() == labels.cpu()).sum()

            accuracy = 100.00 * float(correct) / float(total)

            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(i, loss.data, accuracy))
