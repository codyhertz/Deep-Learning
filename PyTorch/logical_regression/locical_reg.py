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


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        out = self.linear(x)
        return out


input_size = train_dataset[0][0].size()[1] * train_dataset[0][0].size()[2]
output_size = 10

model = LogisticRegressionModel(input_size, output_size)

# loss function
criterion = nn.CrossEntropyLoss()

# optimizer
learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

i = 0
for epoch in range(num_epochs):
    for j, (images, labels) in enumerate(train_loader):
        # load images
        images = Variable(images.view(-1, 28*28))
        labels = Variable(labels)

        # clear grads from last iteration
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward() # get new gradients
        optimizer.step() # updating params

        i += 1

        if i % 500 == 0:
            # calculating accuracay
            correct = 0
            total = 0

            for images, labels in test_loader:
                images = Variable(images.view(-1, 28*28))

                outputs = model(images)

                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)

                correct += (predicted == labels).sum()

            accuracy = 100.00 * float(correct) / float(total)

            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(i, loss.data, accuracy))

torch.save(model.state_dict(), 'awesome_model.pkl')
