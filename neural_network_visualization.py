import numpy
import torch
import torchviz

x_train = numpy.array([[4.7], [2.4], [7.5], [7.1], [4.3], [7.816], [8.9], [5.2], [8.59], [2.1], [8], [10], [4.5], [6], [4]], dtype=numpy.float32)
y_train = numpy.array([[2.6], [1.6], [3.09], [2.4], [2.4], [3.357], [2.6], [1.96], [3.53], [1.76], [3.2], [3.5], [1.6], [2.5], [2.2]], dtype=numpy.float32)
X_train = torch.from_numpy(x_train)
Y_train = torch.from_numpy(y_train)
input_size = 1
hidden_size = 100
output_size = 1
model1 = torch.nn.Sequential(torch.nn.Linear(input_size, hidden_size), torch.nn.Linear(hidden_size, output_size))
y = model1(X_train)
torchviz.make_dot(y.mean(), params=dict(model1.named_parameters())).render('model1', view=True)
model2 = torch.nn.Sequential(torch.nn.Linear(input_size, hidden_size), torch.nn.Linear(hidden_size, hidden_size), torch.nn.Sigmoid(), torch.nn.Linear(hidden_size, output_size))
y = model2(X_train)
torchviz.make_dot(y.mean(), params=dict(model1.named_parameters())).render('model2', view=True)