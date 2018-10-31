import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


class ConvNet(nn.Module):
  def __init__(self, input_channels, output_channels, kernel_size, stride, padding):
    """
    In the constructor we instantiate two nn.Linear modules and assign them as
    member variables.
    """
    super(ConvNet, self).__init__()
    self.layer1 = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=kernel_size, stride=stride, padding=padding)
  def forward(self, x):
    """
    In the forward function we accept a Tensor of input data and we must return
    a Tensor of output data. We can use Modules defined in the constructor as
    well as arbitrary (differentiable) operations on Tensors.
    """
    x = self.layer1(x)
    return x
@profile
def main():

  if (len(sys.argv) < 9):
    print("usage: conv.py <input size> <input channels> <padding> <batch size> <kernel size> <output channels> <stride> <optimization algo>")
    return

  input_size = int(sys.argv[1])
  input_channels = int(sys.argv[2])
  padding = int(sys.argv[3])
  batch_size = int(sys.argv[4])
  kernel_size = int(sys.argv[5])
  output_channels = int(sys.argv[6])
  stride = int(sys.argv[7])
  opt_algo = sys.argv[8]

  print("Running: ", sys.argv[0])
  print("Input size: ", input_size)
  print("Input channels: ", input_channels)
  print("Padding: ", padding)
  print("Batch size: ", batch_size)
  print("Kernel size: ", kernel_size)
  print("Output channels: ", output_channels)
  print("Stride: ", stride)
  print("Opt algo: ", opt_algo)


# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.

# Create random Tensors to hold inputs and outputs
  x = Variable(torch.randn(batch_size, input_channels, input_size, input_size))
  output_size = (input_size - kernel_size + 2*padding)/(stride) + 1
  y = Variable(torch.randn(batch_size, output_channels, output_size, output_size))

# Construct our model by instantiating the class defined above.
  myConv = ConvNet(input_channels, output_channels, kernel_size, stride, padding)

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
  loss_fn = nn.MSELoss(size_average=False)
  if(opt_algo == "SGD"):
    optimizer = optim.SGD(myConv.parameters(), lr=1e-4)
  elif(opt_algo == "Adam"):
    optimizer = optim.Adam(myConv.parameters(), lr=0.001)

# training part to be implemented
  myConv.train()
  optimizer.zero_grad()
  y1 = myConv(x)
  loss = loss_fn(y1, y)
  loss.backward()
  optimizer.step()



# inference part
  myConv.eval()
  y1 = myConv(x)

if __name__ == '__main__':
  main()