import torch
import numpy as np

# create matrix
arr = [[1, 2], [3, 4]]

arr = torch.Tensor(arr)

print(arr)

ones = torch.ones((2, 2))

print(ones)

rand = torch.rand(2, 2)

print(rand)

# seeds for reproducibility
torch.manual_seed(0)
print(torch.rand(2, 2))

torch.manual_seed(0)
print(torch.rand(2, 2))

# numpy to torch
# data type matters
np_array = np.ones((2, 2))

print(type(np_array))

torch_tensor = torch.from_numpy(np_array)

print(type(torch_tensor))

# torch to numpy

torch_to_numpy = torch_tensor.numpy()

print(type(torch_to_numpy))

# cpu vs gpu computation
# initialive as cpu
tensor_cpu = torch.ones(2, 2)
print(tensor_cpu)

# to gpu
tensor_cpu.cuda()
print(tensor_cpu)

#to cpu
tensor_cpu.cpu()
print(tensor_cpu)

# matrix operations

#resize
a = torch.ones(2, 2)
print(a, a.size())

a = a.view(4)
print(a, a.size())

# elemnt wise addition
a = a.view(2, 2) # reshape

b = torch.ones(2, 2)

c = a + b
print(c)

c = torch.add(a, b)
print(c)

# in place addition
c.add_(a)

print(c)

# elemt wise subtraction

print(a - b)
print(a.sub(b)) # not in place
print(a.sub_(b)) # in place

# mult
a = torch.ones(2, 2)
b = torch.zeros(2, 2)
print(a * b)
print(torch.mul(a, b))

print(a.mul_(b))

# division
a = torch.ones(2, 2)
b = torch.zeros(2, 2)
print(b / a)
print(torch.div(b, a))
print(b.div_(a))

# mean(average)
a = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(a.size())
print(a.mean(dim=0))

a = torch.Tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
print(a.size())
print(a.mean(dim=1))

# standard deviation
a = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(a.std(dim=0))
