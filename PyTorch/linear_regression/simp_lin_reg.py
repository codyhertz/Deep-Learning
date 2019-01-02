import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt


x_values = [i for i in range(11)]

x_train = np.array(x_values, dtype=np.float32)
print(x_train.shape)

# reshape, needs to be 2d
x_train = x_train.reshape(-1, 1)
print(x_train.shape)

y_values = [2 * i + 1 for i in x_values]

y_train = np.array(y_values, dtype=np.float32)
print(y_train.shape)

y_train = y_train.reshape(-1, 1)


class LinearRegressionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        out = self.linear(x)
        return out


input_size = 1
output_size = 1

model = LinearRegressionModel(input_size, output_size)

# loss Mean squared error, normal cost function for linear regression
criterion = nn.MSELoss()
learning_rate = 0.01

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 100

for epoch in range(epochs):
    inputs = Variable(torch.from_numpy(x_train))
    labels = Variable(torch.from_numpy(y_train))

    # clear gradients wrt parameters
    optimizer.zero_grad()

    # calls forward
    predicted = model(inputs)

    loss = criterion(predicted, labels)

    # get gradients
    loss.backward()

    # update params
    optimizer.step()

    print('epoch {}, loss {}'.format(epoch + 1, loss.data))

predicted_vals = model(Variable(torch.from_numpy(x_train))).data.numpy()
print(predicted_vals)

# y = 2X + 1
print(y_train)

plt.clf()

plt.plot(x_train, y_train, 'go', label='True data', alpha=0.5)
plt.plot(x_train, predicted_vals, '--', label='Predictions', alpha=0.5)

plt.legend(loc='best')
plt.show()

# saving model(params)
torch.save(model.state_dict(), 'model.pkl')

# load
model.load_state_dict(torch.load('model.pkl'))
