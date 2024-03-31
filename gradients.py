import torch

def calculate(t):
    return t * 2

@torch.no_grad()
def calculate_with_no_grad(t):
    return t * 2

tensor = torch.Tensor([[1, 2, 3], [4, 5, 6]])
tensor.requires_grad_()
result_tensor = calculate(tensor)
print(result_tensor)
print(result_tensor.requires_grad)
result_tensor = calculate_with_no_grad(tensor)
print(result_tensor)
print(result_tensor.requires_grad)
# will always have requires_grad = False
detached_tensor = tensor.detach()
print(detached_tensor.requires_grad)