import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


class FC(nn.Module):
  def __init__(self, in_features, out_features):
    """
    In the constructor we instantiate two nn.Linear modules and assign them as
    member variables.
    """
    super(FC, self).__init__()
    self.layer1 = nn.Linear(in_features = in_features, out_features = out_features)
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

  if (len(sys.argv) < 3):
    print("usage: conv.py <in_features> <out_features>")
    return

  in_features = int(sys.argv[1])
  out_features = int(sys.argv[2])

  print("Running: ", sys.argv[0])
  print("Input features: ", in_features)
  print("Output features: ", out_features)


# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.

# Create random Tensors to hold inputs and outputs
  x = Variable(torch.randn(in_features))
  y = Variable(torch.randn(out_features))

# Construct our model by instantiating the class defined above.
  myConv = FC(in_features, out_features)

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
  loss_fn = nn.MSELoss(size_average=False)

# training part to be implemented
  myConv.train()
  #optimizer.zero_grad()
  y1 = myConv.forward(x)
  loss = loss_fn(y1, y)
  loss.backward()
  #optimizer.step()



# inference part
  myConv.eval()
  y1 = myConv(x)

if __name__ == '__main__':
  main()