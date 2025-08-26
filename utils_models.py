import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from attacks import GetSubnet
import math

class SubnetLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(SubnetLinear, self).__init__(in_features, out_features, bias=True)
        self.popup_scores = Parameter(torch.Tensor(self.weight.shape))
        nn.init.kaiming_uniform_(self.popup_scores, a=math.sqrt(5))
        self.k = 1.0

    def forward(self, x):
        # Ensure all components are on the same device
        adj = GetSubnet.apply(self.popup_scores.abs().to(x.device), self.k)
        self.w = self.weight.to(x.device) * adj
        return F.linear(x, self.w, self.bias.to(x.device))



class SubnetConv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(SubnetConv, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.popup_scores = Parameter(torch.Tensor(self.weight.shape))
        nn.init.kaiming_uniform_(self.popup_scores, a=math.sqrt(5))
        self.k = 1.0

    def forward(self, x):
        adj = GetSubnet.apply(self.popup_scores.abs(), self.k)
        self.w = self.weight * adj
        x = F.conv2d(x, self.w, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x

    def set_k(self, new_k):
        assert 0 <= new_k <= 1, "k value must be between 0 and 1"
        self.k = new_k


# class SimpleCNN(nn.Module):
#     def __init__(self, conv_layers_config, fc_layers_config, img_dim=32):
#         super(SimpleCNN, self).__init__()
#         self.conv_layers = nn.ModuleList()
#         self.fc_layers_config = fc_layers_config  # Store fc_layers_config to initialize later
#         self.fc_layers = None  # Placeholder, will initialize in forward
#
#         # Create the convolutional layers based on the configuration
#         for layer_cfg in conv_layers_config:
#             self.conv_layers.append(SubnetConv(**layer_cfg))
#
#         self.img_dim = img_dim  # Image dimension for dynamic calculation
#
#     def forward(self, x):
#         # Pass input through convolutional layers
#         for conv_layer in self.conv_layers:
#             x = torch.relu(conv_layer(x))
#             x = torch.max_pool2d(x, 2)
#
#         # Flatten the output for fully connected layers
#         x = x.view(x.size(0), -1)
#
#         # Initialize fc_layers dynamically based on the calculated input features
#         if self.fc_layers is None:
#             in_features = x.size(1)  # Calculate in_features from flattened conv layer output
#             self.fc_layers_config[0]['in_features'] = in_features  # Set the first layer's in_features
#
#             # Create the fully connected layers based on the configuration
#             self.fc_layers = nn.ModuleList([SubnetLinear(**cfg) for cfg in self.fc_layers_config])
#
#         # Pass input through fully connected layers
#         for fc_layer in self.fc_layers[:-1]:
#             x = torch.relu(fc_layer(x))
#         x = self.fc_layers[-1](x)  # Last layer without ReLU
#
#         return x
#
#     def set_conv_k(self, new_k):
#         for module in self.conv_layers:
#             if hasattr(module, "set_k"):
#                 module.set_k(new_k)
#
#     def set_lin_k(self, new_k):
#         for module in self.fc_layers:
#             if hasattr(module, "set_k"):
#                 module.set_k(new_k)


# class SimpleCNN(nn.Module):
#     def __init__(self, conv_layers_config, fc_layers_config, img_dim=32):
#         super(SimpleCNN, self).__init__()
#         self.conv_layers = nn.ModuleList()
#         for layer_cfg in conv_layers_config:
#             self.conv_layers.append(SubnetConv(**layer_cfg))
#
#         # 计算第一个全连接层的in_features
#         conv_output_dim = img_dim // 4  # 假设有2个MaxPool层，stride为2
#         fc_layers_config = [
#             {
#                 "in_features": conv_output_dim * conv_output_dim * conv_layers_config[-1]["out_channels"]
#                 if cfg["in_features"] == "computed" else cfg["in_features"],
#                 "out_features": cfg["out_features"]
#             }
#             for cfg in fc_layers_config
#         ]
#
#         self.fc_layers = nn.ModuleList()
#         for layer_cfg in fc_layers_config:
#             self.fc_layers.append(SubnetLinear(**layer_cfg))
#
#     def forward(self, x):
#         for conv_layer in self.conv_layers:
#             x = torch.relu(conv_layer(x))
#             x = torch.max_pool2d(x, 2)
#
#         x = x.view(x.size(0), -1)
#
#         for fc_layer in self.fc_layers[:-1]:
#             x = torch.relu(fc_layer(x))
#         x = self.fc_layers[-1](x)
#
#         return x
#
#     def set_conv_k(self, new_k):
#         for module in self.conv_layers:
#             if hasattr(module, "set_k"):
#                 module.set_k(new_k)
#
#     def set_lin_k(self, new_k):
#         for module in self.fc_layers:
#             if hasattr(module, "set_k"):
#                 module.set_k(new_k)

class TexasFullyConnectedNN(nn.Module):
    def __init__(self, input_dim=6169, hidden_dims=[128, 64], output_dim=100):
        super(TexasFullyConnectedNN, self).__init__()
        layers = []
        in_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(SubnetLinear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.5))  # Optional dropout for regularization
            in_dim = hidden_dim

        layers.append(SubnetLinear(in_dim, output_dim))  # Final output layer
        self.fc_layers = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.fc_layers(x)
        return logits

    def set_k(self, new_k):
        for module in self.fc_layers:
            if hasattr(module, "set_k"):
                module.set_k(new_k)

class SimpleCNN_cifar10(nn.Module):
    def __init__(self, input_channels=3, num_classes=10):
        super(SimpleCNN_cifar10, self).__init__()

        # Define the convolutional layers using SubnetConv
        self.conv1 = SubnetConv(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = SubnetConv(32, 64, kernel_size=3, padding=1)
        self.conv3 = SubnetConv(64, 128, kernel_size=3, padding=1)

        # Define the fully connected layers using SubnetLinear
        self.fc1 = SubnetLinear(128 * 4 * 4, 256)
        self.fc2 = SubnetLinear(256, num_classes)

        # MaxPooling layer
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)

        x = x.view(-1, 128 * 4 * 4)  # Flatten the output for the fully connected layers

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # print(x.shape)
        return x

    def set_conv_k(self, new_k):
        for module in self.modules():
            if isinstance(module, SubnetConv):
                module.k = new_k

    def set_lin_k(self, new_k):
        for module in self.modules():
            if isinstance(module, SubnetLinear):
                module.k = new_k

class SimpleCNN_mnist(nn.Module):
    def __init__(self, input_channels=1, num_classes=10):
        super(SimpleCNN_mnist, self).__init__()

        # Define the convolutional layers using SubnetConv
        self.conv1 = SubnetConv(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = SubnetConv(32, 64, kernel_size=3, padding=1)

        # Define the fully connected layers using SubnetLinear
        self.fc1 = SubnetLinear(64 * 7 * 7, 128)  # Adjust the input size for MNIST (28x28 -> 7x7 after pooling)
        self.fc2 = SubnetLinear(128, num_classes)

        # MaxPooling layer
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = x.view(-1, 64 * 7 * 7)  # Flatten the output for the fully connected layers

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def set_conv_k(self, new_k):
        for module in self.modules():
            if isinstance(module, SubnetConv):
                module.k = new_k

    def set_lin_k(self, new_k):
        for module in self.modules():
            if isinstance(module, SubnetLinear):
                module.k = new_k

class Net(nn.Module):
    def __init__(self, conv_layers_config, fc_layers_config, img_dim=32):
        super(Net, self).__init__()
        self.conv_layers = nn.ModuleList()
        for layer_cfg in conv_layers_config:
            self.conv_layers.append(SubnetConv(**layer_cfg))

        # 计算第一个全连接层的in_features
        conv_output_dim = img_dim // 4  # 假设有2个MaxPool层，stride为2
        fc_layers_config = [
            {
                "in_features": conv_output_dim * conv_output_dim * conv_layers_config[-1]["out_channels"]
                if cfg["in_features"] == "computed" else cfg["in_features"],
                "out_features": cfg["out_features"]
            }
            for cfg in fc_layers_config
        ]

        self.fc_layers = nn.ModuleList()
        for layer_cfg in fc_layers_config:
            self.fc_layers.append(SubnetLinear(**layer_cfg))

    def forward(self, x):
        for conv_layer in self.conv_layers:
            x = torch.relu(conv_layer(x))
            x = torch.max_pool2d(x, 2)

        x = x.view(x.size(0), -1)

        for fc_layer in self.fc_layers[:-1]:
            x = torch.relu(fc_layer(x))
        x = self.fc_layers[-1](x)

        return x

    def set_conv_k(self, new_k):
        for module in self.conv_layers:
            if hasattr(module, "set_k"):
                module.set_k(new_k)

    def set_lin_k(self, new_k):
        for module in self.fc_layers:
            if hasattr(module, "set_k"):
                module.set_k(new_k)


class UnifiedFullyConnectedNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=100):  # Ensure output_dim matches the number of classes
        super(UnifiedFullyConnectedNN, self).__init__()
        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(SubnetLinear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.5))  # Optional dropout for regularization
            in_dim = hidden_dim
        layers.append(SubnetLinear(in_dim, output_dim))  # Ensure output_dim matches the number of classes
        self.fc_layers = nn.Sequential(*layers)

    def forward(self, x):
        # print(f"Input shape: {x.shape}")
        # print(f"output_dim: {self.fc_layers[-1].out_features}")
        logits = self.fc_layers(x)  # Return raw logits
        # print(f"Logits shape: {logits.shape}")
        return logits

    def set_k(self, new_k):
        for module in self.fc_layers:
            if hasattr(module, "set_k"):
                module.set_k(new_k)





# class Net(nn.Module):
#     def __init__(self, conv_layers_config, fc_layers_config, img_dim=32):
#         super(Net, self).__init__()
#         self.conv_layers = nn.ModuleList()
#         for layer_cfg in conv_layers_config:
#             self.conv_layers.append(SubnetConv(**layer_cfg))
#
#         # 手动设置第一个全连接层的in_features
#         fc_layers_config[0]['in_features'] = 400  # 根据调试结果修正为400
#
#         self.fc_layers = nn.ModuleList()
#         for layer_cfg in fc_layers_config:
#             self.fc_layers.append(SubnetLinear(**layer_cfg))
#
#     def forward(self, x):
#         # print(f"Input shape: {x.shape}")  # 调试输出
#
#         for conv_layer in self.conv_layers:
#             x = torch.relu(conv_layer(x))
#             x = torch.max_pool2d(x, 2)
#             # print(f"After conv and pool: {x.shape}")  # 调试输出
#
#         x = x.view(x.size(0), -1)
#         # print(f"After flatten: {x.shape}")  # 调试输出
#
#         for fc_layer in self.fc_layers[:-1]:
#             x = torch.relu(fc_layer(x))
#             # print(f"After fc_layer: {x.shape}")  # 调试输出
#         x = self.fc_layers[-1](x)
#
#         return x
#
#     def set_conv_k(self, new_k):
#         for module in self.conv_layers:
#             if hasattr(module, "set_k"):
#                 module.set_k(new_k)
#
#     def set_lin_k(self, new_k):
#         for module in self.fc_layers:
#             if hasattr(module, "set_k"):
#                 module.set_k(new_k)


def apply_min_activation_attack(x, dropout_rate=0.5):
    """
    对输入 x 应用最大激活攻击，随机丢弃激活值最高的神经元。
    :param x: 输入张量
    :param dropout_rate: 丢弃的神经元比例，介于 0 和 1 之间
    :return: 经过最大激活攻击后的张量
    """
    if dropout_rate > 0 and dropout_rate < 1:
        # 计算每个神经元的平均激活值
        activations = torch.mean(x, dim=0)

        # 找到激活值的分位数阈值
        threshold = torch.quantile(activations, 1 - dropout_rate)

        # 生成一个掩码，只保留激活值低于阈值的神经元
        mask = activations <= threshold

        # 应用掩码，将不符合条件的神经元置零
        x = x * mask.float()

    return x

def apply_sample_dropping_attack(x, dropout_rate=0.5):
    """
    对输入 x 应用样本丢弃攻击，随机丢弃部分样本。
    :param x: 输入张量
    :param dropout_rate: 丢弃的样本比例，介于 0 和 1 之间
    :return: 经过样本丢弃攻击后的张量
    """
    if dropout_rate > 0 and dropout_rate < 1:
        mask = torch.rand(x.size(0)) > dropout_rate
        x = x[mask]
    return x

def apply_neuron_separation_attack(x, separation_factor=2.0):
    """
    对输入 x 应用神经元分离攻击，增强激活值之间的差异。
    :param x: 输入张量
    :param separation_factor: 分离因子，控制激活值的差异程度
    :return: 经过神经元分离攻击后的张量
    """
    mean_activation = torch.mean(x, dim=0)
    x = x * (mean_activation * separation_factor)
    return x
class AttackedModel(nn.Module):
    def __init__(self, original_model, attack_type, dropout_rate=0.5, separation_factor=2.0):
        super(AttackedModel, self).__init__()
        self.original_model = original_model
        self.attack_type = attack_type
        self.dropout_rate = dropout_rate
        self.separation_factor = separation_factor

    def forward(self, x, attack_enabled=False):
        # Pass through convolutional layers
        if hasattr(self.original_model, 'conv1'):
            x = torch.relu(self.original_model.conv1(x))
            x = self.original_model.pool(x)
        if hasattr(self.original_model, 'conv2'):
            x = torch.relu(self.original_model.conv2(x))
            x = self.original_model.pool(x)
        if hasattr(self.original_model, 'conv3'):
            x = torch.relu(self.original_model.conv3(x))
            x = self.original_model.pool(x)

        # Apply the specified attack
        if self.attack_type == 'min_activation' and attack_enabled:
            x = apply_min_activation_attack(x, self.dropout_rate)
        elif self.attack_type == 'sample_dropping' and attack_enabled:
            x = apply_sample_dropping_attack(x, self.dropout_rate)
        elif self.attack_type == 'neuron_separation' and attack_enabled:
            x = apply_neuron_separation_attack(x, self.separation_factor)
        else:
            x = x

        # Flatten and pass through fully connected layers
        x = x.view(x.size(0), -1)
        x = torch.relu(self.original_model.fc1(x))
        x = self.original_model.fc2(x)

        return x

    def load_state_dict(self, state_dict, strict=True):
        """
        Load the state_dict into the original model inside AttackedModel.
        """
        self.original_model.load_state_dict(state_dict, strict=strict)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """
        Return the state_dict of the original model inside AttackedModel.
        """
        return self.original_model.state_dict(destination, prefix, keep_vars)

#
# class AttackedModel(nn.Module):
#     def __init__(self, original_model, attack_type, dropout_rate=0.5, separation_factor=2.0):
#         super(AttackedModel, self).__init__()
#         self.original_model = original_model
#         self.attack_type = attack_type
#         self.dropout_rate = dropout_rate
#         self.separation_factor = separation_factor
#
#     def forward(self, x, attack_enabled=False):
#         # Check if the original model has convolutional layers and apply them
#         if hasattr(self.original_model, 'conv_layers') and isinstance(self.original_model.conv_layers, nn.ModuleList):
#             for conv_layer in self.original_model.conv_layers:
#                 x = torch.relu(conv_layer(x))
#                 x = torch.max_pool2d(x, 2)
#
#         # Apply the specified attack
#         if self.attack_type == 'min_activation' and attack_enabled:
#             x = apply_min_activation_attack(x, self.dropout_rate)
#         elif self.attack_type == 'sample_dropping' and attack_enabled:
#             x = apply_sample_dropping_attack(x, self.dropout_rate)
#         elif self.attack_type == 'neuron_separation' and attack_enabled:
#             x = apply_neuron_separation_attack(x, self.separation_factor)
#         elif attack_enabled:
#             raise ValueError(f"Unknown attack type: {self.attack_type}")
#
#         # Flatten and pass through fully connected layers
#         x = x.view(x.size(0), -1)
#         for fc_layer in self.original_model.fc_layers[:-1]:
#             x = torch.relu(fc_layer(x))
#         x = self.original_model.fc_layers[-1](x)
#
#         return x
#
#     def load_state_dict(self, state_dict, strict=True):
#         """
#         Load the state_dict into the original model inside AttackedModel.
#         """
#         self.original_model.load_state_dict(state_dict, strict=strict)
#
#     def state_dict(self, destination=None, prefix='', keep_vars=False):
#         """
#         Return the state_dict of the original model inside AttackedModel.
#         """
#         return self.original_model.state_dict(destination, prefix, keep_vars)



