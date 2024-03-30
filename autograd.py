import torch

tensor1 = torch.Tensor([[1, 2, 3], [4, 5, 6]])
tensor2 = torch.Tensor([[4, 5, 6], [7, 8, 9]])
print(tensor1.requires_grad)
tensor1.requires_grad_()
print(tensor1.requires_grad)
print(tensor1.grad)
print(tensor1.grad_fn)
output_tensor = tensor1 * tensor2
print(output_tensor.requires_grad)
print(output_tensor.grad_fn)
output_tensor = (tensor1 * tensor2).mean()
print(output_tensor.requires_grad)
# references the last function used, in this case mean
print(output_tensor.grad_fn)
output_tensor.retain_grad()
output_tensor.backward()
print(output_tensor.grad)
print(tensor1.grad)
print(tensor2.grad)
print(output_tensor.grad)