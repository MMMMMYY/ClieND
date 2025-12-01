import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.nn.parameter import Parameter
import torch
import math

class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        # Get the subnetwork by sorting the scores and using the top k%
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int((1 - k) * scores.numel())

        # flat_out and out access the same memory.
        flat_out = out.flatten()
        # print("connections: {}".format(flat_out.shape))
        # print(flat_out)
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1

        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None

class SubnetConv(nn.Conv2d):
    # self.k is the % of weights remaining, a real number in [0,1]
    # self.popup_scores is a Parameter which has the same shape as self.weight
    # Gradients to self.weight, self.bias have been turned off by default.

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(SubnetConv, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        self.popup_scores = Parameter(torch.Tensor(self.weight.shape))
        nn.init.kaiming_uniform_(self.popup_scores, a=math.sqrt(5))

        # self.weight.requires_grad = False
        # if self.bias is not None:
        #     self.bias.requires_grad = False
        self.w = 0

    def forward(self, x):
        # Get the subnetwork by sorting the scores.
        adj = GetSubnet.apply(self.popup_scores.abs(), 1)
        # Use only the subnetwork in the forward pass.
        self.w = self.weight * adj
        x = F.conv2d(
            x, self.w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x

__all__ = ['mnist_cnn']


def mnist_cnn(**kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    """
    model = SimpleCNNMNIST(**kwargs)
    return model

class SimpleCNNMNIST(nn.Module):
    def __init__(self, cov_layer, input_dim, hidden_dims, output_dim=10):
        super(SimpleCNNMNIST, self).__init__()
        self.conv1 = cov_layer(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = cov_layer(6, 16, 5)

        # for now, we hard coded this network
        # i.e. we fix the number of hidden layers i.e. 2 layers
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x