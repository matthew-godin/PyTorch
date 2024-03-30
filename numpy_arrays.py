import numpy
import torch

tensor = torch.rand(4, 3)
print(tensor)
print(type(tensor))
# the numpy array and torch tensor share the same underlying memory
numpy_from_tensor = tensor.numpy()
print(numpy_from_tensor)
print(type(numpy_from_tensor))
tensor_from_numpy = torch.from_numpy(numpy_from_tensor)
print(tensor_from_numpy)
print(type(tensor_from_numpy))
numpy_array = numpy.array([4, 8])
# shares same memory if numpy array created on same device
tensor_from_numpy_array = torch.as_tensor(numpy_array)
# makes a copy and does not share same memory
tensor_copy_from_numpy_array = torch.tensor(numpy_array)
numpy_array[0] = 22
print(tensor_from_numpy_array)
print(tensor_copy_from_numpy_array)