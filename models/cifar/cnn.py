import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.nn.parameter import Parameter
import torch
import math

def mask_top_k_percent(tensor, k):
    assert 0 < k <= 100, "k must be in the range (0, 100]"

    flat_tensor = tensor.view(-1)
    num_elements = flat_tensor.numel()
    num_to_keep = int(num_elements * (k / 100)) # number of weights to keep, k is the percentage of weights remaining

    _, indices = flat_tensor.abs().sort(descending=True)
    top_k_indices = indices[:num_to_keep]

    mask = torch.zeros_like(flat_tensor)
    mask[top_k_indices] = 1

    mask = mask.view(tensor.shape)
    masked_tensor = tensor * mask

    return masked_tensor

class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):

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

class SubnetLinear(nn.Linear):
    # self.k is the % of weights remaining, a real number in [0,1]
    # self.popup_scores is a Parameter which has the same shape as self.weight
    # Gradients to self.weight, self.bias have been turned off.

    def __init__(self, in_features, out_features, bias=True):
        super(SubnetLinear, self).__init__(in_features, out_features, bias=True)
        self.popup_scores = Parameter(torch.Tensor(self.weight.shape))
        nn.init.kaiming_uniform_(self.popup_scores, a=math.sqrt(5))
        self.w = 0
        # self.register_buffer('w', None)


    def forward(self, x):
        # Get the subnetwork by sorting the scores.
        adj = GetSubnet.apply(self.popup_scores.abs(), 1)

        # Use only the subnetwork in the forward pass.
        self.w = self.weight * adj
        x = F.linear(x, self.w, self.bias)

        return x
    
    def set_k(self, new_k):
        """
        Set the value of the k parameter.
        Args:
            new_k (float): The new value of k.
        """
        assert 0 <= new_k <= 1, "k value must be between 0 and 1"
        self.k = new_k

    def mask_top_k_weights(self, k):
        self.weight.data = mask_top_k_percent(self.weight.data, k)
    

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
        self.k = 1

    def forward(self, x):
        # Get the subnetwork by sorting the scores.
        # print(self.k)
        adj = GetSubnet.apply(self.popup_scores.abs(), self.k)
        # print(adj.shape)
        # Use only the subnetwork in the forward pass.
        self.w = self.weight * adj
        x = F.conv2d(
            x, self.w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x

    def set_k(self, new_k):
        """
        Set the value of the k parameter.
        Args:
            new_k (float): The new value of k.
        """
        assert 0 <= new_k <= 1, "k value must be between 0 and 1"
        self.k = new_k

    def mask_top_k_weights(self, k):
        self.weight.data = mask_top_k_percent(self.weight.data, k)



__all__ = ['cnn']

def cnn(**kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    """
    model = SimpleCNN(**kwargs)
    return model

def cnn_mnist(**kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    """
    model = SimpleCNNMNIST(**kwargs)
    return model

# class SimpleCNN(nn.Module):
#     def __init__(self, cov_layer, lin_layer, input_dim, hidden_dims, output_dim=10):
#         super(SimpleCNN, self).__init__()
#         self.conv1 = cov_layer(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = cov_layer(6, 16, 5)
#
#         # for now, we hard coded this network
#         # i.e. we fix the number of hidden layers i.e. 2 layers
#         self.fc1 = lin_layer(input_dim, hidden_dims[0])
#         self.fc2 = lin_layer(hidden_dims[0], hidden_dims[1])
#         self.fc3 = lin_layer(hidden_dims[1], output_dim)
#
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
#
#     def set_conv_k(self, new_k):
#         """
#         Set the value of the k parameter for convolutional layers.
#         Args:
#             new_k (float): The new value of k.
#         """
#         for module in [self.conv1, self.conv2]:
#             if hasattr(module, "set_k"):
#                 module.set_k(new_k)
#
#     def set_lin_k(self, new_k):
#         """
#         Set the value of the k parameter for linear layers.
#         Args:
#             new_k (float): The new value of k.
#         """
#         for module in [self.fc1, self.fc2, self.fc3]:
#             if hasattr(module, "set_k"):
#                 module.set_k(new_k)
#
#     def mask_top_k_weights(self, k):
#         for module in [self.conv1, self.conv2, self.fc1, self.fc2, self.fc3]:
#                     if hasattr(module, "mask_top_k_weights"):
#                         module.mask_top_k_weights(k)
#
#     def predict_internal_states(self, x):
#         internal_states = []
#         x = F.relu(self.conv1(x))
#         internal_states.append(x.clone().detach())
#         x = self.pool(x)
#         x = F.relu(self.conv2(x))
#         internal_states.append(x.clone().detach())
#         x = self.pool(x)
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         internal_states.append(x.clone().detach())
#         x = F.relu(self.fc2(x))
#         internal_states.append(x.clone().detach())
#         x = self.fc3(x)
#         internal_states.append(x.clone().detach())
#         return internal_states


class SimpleCNN(nn.Module):
    def __init__(self, conv_layer, lin_layer, input_dim, hidden_dims, output_dim=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = conv_layer(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = conv_layer(6, 16, 5)

        if not isinstance(hidden_dims, list) or len(hidden_dims) < 1:
            raise ValueError("hidden_dims must be a list with at least one integer")

        self.fc1 = lin_layer(16 * 5 * 5, hidden_dims[0] // 2)
        self.fc2 = lin_layer(hidden_dims[0] // 2, output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
#
#     def set_conv_k(self, new_k):
#         """
#         Set the value of the k parameter for convolutional layers.
#         Args:
#             new_k (float): The new value of k.
#         """
#         for module in [self.conv1, self.conv2]:
#             if hasattr(module, "set_k"):
#                 module.set_k(new_k)
#
#     def set_lin_k(self, new_k):
#         """
#         Set the value of the k parameter for linear layers.
#         Args:
#             new_k (float): The new value of k.
#         """
#         for module in [self.fc1, self.fc2]:
#             if hasattr(module, "set_k"):
#                 module.set_k(new_k)
#
#     def mask_top_k_weights(self, k):
#         for module in [self.conv1, self.conv2, self.fc1, self.fc2]:
#             if hasattr(module, "mask_top_k_weights"):
#                 module.mask_top_k_weights(k)
#
#     def predict_internal_states(self, x):
#         internal_states = []
#         x = F.relu(self.conv1(x))
#         internal_states.append(x.clone().detach())
#         x = self.pool(x)
#         x = F.relu(self.conv2(x))
#         internal_states.append(x.clone().detach())
#         x = self.pool(x)
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         internal_states.append(x.clone().detach())
#         x = self.fc2(x)
#         internal_states.append(x.clone().detach())
#         return internal_states



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