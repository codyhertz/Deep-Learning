import torch
from torch.autograd import Variable

a = Variable(torch.ones(2, 2), requires_grad=True)
print(a)

b = Variable(torch.ones(2, 2), requires_grad=True)
print(a + b)

# similiar to tensors
# variables accumulate gradients

x = Variable(torch.ones(2), requires_grad=True)
y = 5 * (x + 1) ** 2
print(x, y)

o = .5 * torch.sum(y) # scaler value
print(o)

o.backward()
print(x.grad)
