import torch
import pandas as pd
import logging
import os
import sys
from data_reader import DataReader
from constants import *
from aggregator import *
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.nn.parameter import Parameter
import math

def make_logger(name, save_dir, save_filename):
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", datefmt=DATE_FORMAT)
    ch.setFormatter(formatter)

    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, save_filename + ".txt"), mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def select_by_threshold(to_share: torch.Tensor, select_fraction: float, select_threshold: float = 1):
    """
    Apply the privacy-preserving method following selection-by-threshold approach
    """
    threshold_count = round(to_share.size(0) * select_threshold)
    selection_count = round(to_share.size(0) * select_fraction)
    indices = to_share.topk(threshold_count).indices
    perm = torch.randperm(threshold_count).to(DEVICE)
    indices = indices[perm[:selection_count]]
    rei = torch.zeros(to_share.size()).to(DEVICE)
    rei[indices] = to_share[indices].to(DEVICE)
    to_share = rei.to(DEVICE)
    return to_share, indices


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

class SimpleCNN(nn.Module):
    def __init__(self, cov_layer, lin_layer, input_dim, hidden_dims, output_dim=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = cov_layer(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = cov_layer(6, 16, 5)

        # for now, we hard coded this network
        # i.e. we fix the number of hidden layers i.e. 2 layers
        self.fc1 = lin_layer(input_dim, hidden_dims[0])
        self.fc2 = lin_layer(hidden_dims[0], hidden_dims[1])
        self.fc3 = lin_layer(hidden_dims[1], output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def set_conv_k(self, new_k):
        """
        Set the value of the k parameter for convolutional layers.
        Args:
            new_k (float): The new value of k.
        """
        for module in [self.conv1, self.conv2]:
            if hasattr(module, "set_k"):
                module.set_k(new_k)


class CNN(torch.nn.Module):
    def __init__(self, cov_layer, lin_layer):
        super(CNN, self).__init__()
        self.input_layer = torch.nn.Sequential(
            lin_layer(128 * 4 * 4, 1024),
            torch.nn.ReLU()
        )
        self.hidden_layer = torch.nn.Sequential(
            lin_layer(1024, 256),
            torch.nn.ReLU()
        )
        self.output_layer = torch.nn.Sequential(
            lin_layer(256, 100)
        )

    def forward(self, x):
        out = self.input_layer(x)
        out = self.hidden_layer(out)
        out = self.output_layer(out)
        return out


class ModelPurchase100(torch.nn.Module):
    """
    The model handling purchase-100 data set
    """

    def __init__(self):
        super(ModelPurchase100, self).__init__()
        self.input_layer = torch.nn.Sequential(
            torch.nn.Linear(600, 1024),
            torch.nn.ReLU()
        )
        self.hidden_layer = torch.nn.Sequential(
            torch.nn.Linear(1024, 256),
            torch.nn.ReLU()
        )
        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(256, 100)
        )

    def forward(self, x):
        out = self.input_layer(x)
        out = self.hidden_layer(out)
        out = self.output_layer(out)
        return out


class ModelPreTrainedCIFAR100(torch.nn.Module):
    """
    The model to support pre-trained CIFAR-10 data set
    """

    def __init__(self, cov_layer, lin_layer):
        super(ModelPreTrainedCIFAR100, self).__init__()
        self.input_layer = torch.nn.Sequential(
            lin_layer(64, 1024),
            torch.nn.ReLU()
        )
        self.output_layer = torch.nn.Sequential(
            lin_layer(1024, 128),
            torch.nn.ReLU(),
            lin_layer(128, 10)
        )

    def forward(self, x):
        out = self.input_layer(x)
        out = self.output_layer(out)
        return out


class ModelLocation30(torch.nn.Module):
    """
    The model to handel Location100 dataset
    """

    def __init__(self):
        super(ModelLocation30, self).__init__()
        self.input_layer = torch.nn.Sequential(
            torch.nn.Linear(446, 512),
            torch.nn.ReLU(),
        )
        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(512, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 30),
        )

    def forward(self, x):
        out = self.input_layer(x)
        out = self.output_layer(out)
        return out


class ModelTexas100(torch.nn.Module):
    """
    The model to handel Texas10 dataset
    """

    def __init__(self):
        super(ModelTexas100, self).__init__()
        self.input_layer = torch.nn.Sequential(
            torch.nn.Linear(6169, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU()
        )
        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 100)
        )

    def forward(self, x):
        out = self.input_layer(x)
        out = self.output_layer(out)
        return out


class ModelMNIST(torch.nn.Module):
    """
    The model handling MNIST dataset
    """

    def __init__(self,cov_layer, lin_layer):
        super(ModelMNIST, self).__init__()
        self.input_layer = torch.nn.Sequential(
            lin_layer(784, 1024),
            torch.nn.ReLU(),
            lin_layer(1024, 512),
            torch.nn.ReLU()
        )
        self.output_layer = torch.nn.Sequential(
            lin_layer(512, 128),
            torch.nn.ReLU(),
            lin_layer(128, 10)
        )

    def forward(self, x):
        out = self.input_layer(x)
        out = self.output_layer(out)
        return out


class ModelGnome(torch.nn.Module):
    """
    The model handling Gnome dataset
    """

    def __init__(self):
        super(ModelGnome, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(5547, 25),
            torch.nn.ReLU(),
            torch.nn.Linear(25, 5)
        )

    def forward(self, x):
        return self.network(x)


class TargetModel:
    """
    The model to attack against, the target for attacking
    """

    def __init__(self, data_reader: DataReader, participant_index=0, model=DEFAULT_SET):
        # initialize the model
        if model == PURCHASE100:
            self.model = ModelPurchase100()
        elif model == CIFAR_10:
            self.model = ModelPreTrainedCIFAR100(SubnetConv, SubnetLinear)
        elif model == LOCATION30:
            self.model = ModelLocation30()
        elif model == TEXAS100:
            self.model = ModelTexas100()
        elif model == MNIST:
            self.model = ModelMNIST(SubnetConv, SubnetLinear)
        elif model == GNOME:
            self.model = ModelGnome()
        elif model == CIFAR_100:
            self.model = SimpleCNN(SubnetConv, SubnetLinear, 16 * 5 * 5, [120],10)
        else:
            raise NotImplementedError("Model not supported")

        self.model = self.model.to(DEVICE)

        # initialize the data
        self.test_set = None
        self.train_set = None
        self.last_train_batch = None
        ## for 1-30 clients
        for i in range(1, 30):
            exec("self.train_set%s=None" % i)
            exec("self.train_set_last_batch%s=None" % i)
        ## to do:1. finish last batch update 2. finish batch update

        self.data_reader = data_reader
        self.participant_index = participant_index
        self.load_data()
        # if self.participant_index == 0:
        #     print(self.train_set)
        #     self.load_last_batch()

        # initialize the loss function and optimizer
        self.loss_function = torch.nn.CrossEntropyLoss().to(DEVICE)
        self.optimizer = torch.optim.Adam(self.model.parameters())

        # Initialize recorder
        self.train_loss = -1
        self.train_acc = -1

        # Initialize confidence recorder
        self.mask = torch.ones(BATCH_SIZE)
        self.defend = False
        self.defend_count_down = 0
        self.defend_loss_checker = self.train_loss
        self.drop_out = BATCH_SIZE // 4

    def load_last_batch(self):
        self.last_train_batch = self.data_reader.get_last_train_batch().to(DEVICE)
        self.last_test_batch = self.data_reader.get_last_test_batch().to(DEVICE)

    def load_data(self):
        """
        Load batch indices from the data reader
        :return: None
        """

        self.train_set = self.data_reader.get_train_set(self.participant_index).to(DEVICE)
        self.test_set = self.data_reader.get_test_set(self.participant_index).to(DEVICE)
        # self.last_train_batch = self.data_reader.get_last_train_batch().to(DEVICE)
        # self.last_test_batch = self.data_reader.get_last_test_batch().to(DEVICE)

    def activate_defend(self):
        """
        Activate defend function for this participant
        """
        self.defend = True
        self.defend_count_down = 5
        self.generate_mask()
        self.defend_loss_checker = self.train_loss

    def normal_epoch(self, print_progress=False, by_batch=BATCH_TRAINING):
        """
        Train a normal epoch with the given dataset
        :param print_progress: if print the training progress or not
        :param by_batch: True to train by batch, False otherwise
        :return: the training accuracy and the training loss value
        """
        if self.defend:
            if self.defend_count_down > 0:
                self.defend_count_down -= 1
            else:
                self.generate_mask()
                self.defend_loss_checker = self.train_loss
                self.defend_count_down = 5
        train_loss = 0
        train_acc = 0
        batch_counter = 0
        if by_batch:
            # if self.participant_index == 0:
            #     self.train_set = self.data_reader.get_train_set(0)
            #     print(self.train_set)
            for batch_indices in self.train_set:
                batch_counter += 1
                if print_progress and batch_counter % 100 == 0:
                    print("Currently training for batch {}, overall {} batches"
                          .format(batch_counter, self.train_set.size(0)))
                if self.defend:
                    batch_indices = batch_indices[self.mask == 1]
                # print(batch_indices.type(torch.int64))
                # print(batch_indices)
                batch_x, batch_y = self.data_reader.get_batch(batch_indices.type(torch.int64))
                batch_x = batch_x.to(DEVICE)
                batch_y = batch_y.to(DEVICE)
                out = self.model(batch_x).to(DEVICE)
                # print("The size of output = {}, the size of label = {}".format(out.size(), batch_y.size()))
                # print(batch_y)
                # print(out)
                batch_loss = self.loss_function(out, batch_y)
                train_loss += batch_loss.item()
                prediction = torch.max(out, 1).indices.to(DEVICE)
                # if self.defend:
                #     confidence = torch.nn.Softmax(1)(out)
                #     # confidence = out
                #     max_conf = torch.max(confidence, 1).values.to(DEVICE)
                #     for i in range(confidence.size(0)):
                #         sample = batch_indices[i]
                #         sample = sample.item()
                #         self.record_confidence(sample, max_conf[i])
                batch_acc = (prediction == batch_y).sum().to(DEVICE)
                train_acc += batch_acc.item()
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()
                # print(train_acc)
                # print(batch_counter)
            # self.last_train_batch = torch.tensor([])
            # print(self.last_train_batch)
            if self.last_train_batch != None and len(self.last_train_batch) != 0:
                # print(self.last_train_batch)
                batch_x, batch_y = self.data_reader.get_batch(self.last_train_batch.type(torch.int64))
                batch_x = batch_x.to(DEVICE)
                batch_y = batch_y.to(DEVICE)
                out = self.model(batch_x).to(DEVICE)
                # print("The size of output = {}, the size of label = {}".format(out.size(), batch_y.size()))
                # print(batch_y)
                # print(out)
                batch_loss = self.loss_function(out, batch_y)
                train_loss += batch_loss.item()
                prediction = torch.max(out, 1).indices.to(DEVICE)
                # if self.defend:
                #     confidence = torch.nn.Softmax(1)(out)
                #     # confidence = out
                #     max_conf = torch.max(confidence, 1).values.to(DEVICE)
                #     for i in range(confidence.size(0)):
                #         sample = batch_indices[i]
                #         sample = sample.item()
                #         self.record_confidence(sample, max_conf[i])
                batch_acc = (prediction == batch_y).sum().to(DEVICE)
                # print(batch_acc)
                train_acc += batch_acc.item()
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()
                print("last batch")
        else:
            batch_x, batch_y = self.data_reader.get_batch(self.train_set)
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            # print(batch_x[:10])
            # print(batch_y[:10])
            out = self.model(batch_x[:10]).to(DEVICE)
            # print("The size of output = {}, the size of label = {}".format(out.size(), batch_y.size()))
            batch_loss = self.loss_function(out, batch_y)
            train_loss += batch_loss.item()
            prediction = torch.max(out, 1).indices.to(DEVICE)
            batch_acc = (prediction == batch_y).sum()
            train_acc += batch_acc.item()
            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()
        if self.last_train_batch != None:
            # print(self.train_set.flatten().size(0)+self.last_train_batch.flatten().size(0))
            # print(self.train_acc)
            self.train_acc = train_acc / ((self.train_set.flatten().size(0) + self.last_train_batch.flatten().size(0)))
            self.train_loss = train_loss / (
            (self.train_set.flatten().size(0) + self.last_train_batch.flatten().size(0)))
        else:
            # print(self.train_set.flatten().size(0))
            # print(self.train_acc)
            self.train_acc = train_acc / (self.train_set.flatten().size(0))
            self.train_loss = train_loss / (self.train_set.flatten().size(0))
        if print_progress:
            print("Epoch complete for participant {}, train acc = {}, train loss = {}"
                  .format(self.participant_index, train_acc, train_loss))
        return self.train_loss, self.train_acc

    def test_outcome(self, by_batch=BATCH_TRAINING):
        """
        Test through the test set to get loss value and accuracy
        :return: the test accuracy and test loss value
        """
        test_loss = 0
        test_acc = 0
        if by_batch:
            for batch_indices in self.test_set:
                batch_x, batch_y = self.data_reader.get_batch(batch_indices.type(torch.int64))
                batch_x = batch_x.to(DEVICE)
                batch_y = batch_y.to(DEVICE)
                # print(batch_x)
                with torch.no_grad():
                    out = self.model(batch_x).to(DEVICE)
                    batch_loss = self.loss_function(out, batch_y).to(DEVICE)
                    test_loss += batch_loss.item()
                    prediction = torch.max(out, 1).indices.to(DEVICE)
                    batch_acc = (prediction == batch_y).sum().to(DEVICE)
                    test_acc += batch_acc.item()
        else:
            batch_x, batch_y = self.data_reader.get_batch(self.test_set.type(torch.int64))
            with torch.no_grad():
                out = self.model(batch_x)
                batch_loss = self.loss_function(out, batch_y)
                test_loss += batch_loss.item()
                prediction = torch.max(out, 1).indices
                batch_acc = (prediction == batch_y).sum()
                test_acc += batch_acc.item()
        test_acc = test_acc / (self.test_set.flatten().size(0))
        test_loss = test_loss / (self.test_set.flatten().size(0))
        return test_loss, test_acc

    def test_outcome_by_class(self):
        """
        Test through the test set to get loss value and accuracy
        :return: the test accuracy and test loss value
        """
        test_loss = 0
        test_acc = 0
        class_acc_dict = {}
        class_num_dict = {}
        for batch_indices in self.test_set:
            batch_x, batch_y = self.data_reader.get_batch(batch_indices.type(torch.int64))
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            # batch_x, batch_y = self.data_reader.get_batch(self.test_set.type(torch.int64))
            with torch.no_grad():
                out = self.model(batch_x)
                batch_loss = self.loss_function(out, batch_y)
                test_loss += batch_loss.item()
                prediction = torch.max(out, 1).indices.to(DEVICE)
                # print(prediction)
                # print(batch_y)
                for i, prediction_i in enumerate(prediction):
                    prediction_i = int(prediction_i)
                    batch_y_i = int(batch_y[i])
                    # print("y",batch_y_i)
                    # print("predicted_y",prediction_i)
                    if prediction_i == batch_y_i:
                        if batch_y_i in class_acc_dict.keys():
                            class_acc_dict[batch_y_i] = class_acc_dict[batch_y_i] + 1
                            # print(class_acc_dict)
                        else:
                            class_acc_dict[batch_y_i] = 1
                        # print(class_acc_dict)
                    if batch_y_i in class_num_dict:
                        class_num_dict[batch_y_i] = class_num_dict[batch_y_i] + 1
                    else:
                        class_num_dict[batch_y_i] = 1
        print(class_acc_dict)
        # print(stop)
        class_acc_dict = {k: v / class_num_dict[k] for k, v in class_acc_dict.items()}
        return class_acc_dict

    def get_flatten_parameters(self):
        """
        Return the flatten parameter of the current model
        :return: the flatten parameters as tensor
        """
        out = torch.zeros(0).to(DEVICE)
        with torch.no_grad():
            for parameter in self.model.parameters():
                out = torch.cat([out, parameter.flatten()]).to(DEVICE)
        return out

    def load_parameters(self, parameters: torch.Tensor):
        """
        Load parameters to the current model using the given flatten parameters
        :param parameters: The flatten parameter to load
        :return: None
        """
        start_index = 0
        for param in self.model.parameters():
            length = len(param.flatten())
            to_load = parameters[start_index: start_index + length].to(DEVICE)
            to_load = to_load.reshape(param.size()).to(DEVICE)
            with torch.no_grad():
                param.copy_(to_load).to(DEVICE)
            start_index += length

    def get_epoch_gradient(self, apply_gradient=True):
        """
        Get the gradient for the current epoch
        :param apply_gradient: if apply the gradient or not
        :return: the tensor contains the gradient
        """
        cache = self.get_flatten_parameters().to(DEVICE)
        self.normal_epoch()
        gradient = self.get_flatten_parameters() - cache.to(DEVICE)
        if not apply_gradient:
            self.load_parameters(cache)
        return gradient

    def init_parameters(self, mode=INIT_MODE):
        """
        Initialize the parameters according to given mode
        :param mode: the mode to init with
        :return: None
        """
        if mode == NORMAL:
            to_load = torch.randn(self.get_flatten_parameters().size())
            self.load_parameters(to_load)
        elif mode == UNIFORM:
            to_load = torch.rand(self.get_flatten_parameters().size())
            self.load_parameters(to_load)
        elif mode == PYTORCH_INIT:
            return
        else:
            raise ValueError("Invalid initialization mode")

    def test_gradients(self, gradient: torch.Tensor):
        """
        Make use of the given gradients to run a test, then revert back to the previous status
        """
        cache = self.get_flatten_parameters().to(DEVICE)
        test_param = cache + gradient.to(DEVICE)
        self.load_parameters(test_param)
        loss, acc = self.test_outcome()
        self.load_parameters(cache)
        return loss, acc

    def get_gradzero(self, revert=True):
        validation_data, validation_label = self.data_reader.get_batch(self.data_reader.fl_trust.type(torch.int64))
        print(validation_label)
        cache = self.get_flatten_parameters()
        print(cache)
        out = self.model(validation_data)
        loss = self.loss_function(out, validation_label)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        gradient = self.get_flatten_parameters() - cache
        # gradient, indices = select_by_threshold(gradient, GRADIENT_EXCHANGE_RATE, GRADIENT_SAMPLE_THRESHOLD)
        # grad_zero = self.get_flatten_parameters()
        if revert:
            self.load_parameters(cache)
        return gradient


class FederatedModel(TargetModel):
    """
    Representing the class of federated learning members
    """

    def __init__(self, reader: DataReader, aggregator: Aggregator, participant_index=0):
        super(FederatedModel, self).__init__(reader, participant_index)
        self.aggregator = aggregator
        self.member_list = []
        self.nonmember_list = []

    # def get_attacker_sample(self):
    #     return self.BlackBoxMalicious.attacker_sample()

    def update_aggregator(self, aggregator):
        self.aggregator = aggregator.to(DEVICE)

    def get_aggregator(self):
        return self.aggregator

    def init_global_model(self):
        """
        Initialize the current model as the global model
        :return: None
        """
        self.init_parameters()
        if DEFAULT_DISTRIBUTION is None:
            self.test_set = self.data_reader.test_set.to(DEVICE)
        elif DEFAULT_DISTRIBUTION == CLASS_BASED:
            length = self.data_reader.test_set.size(0)
            length -= length % BATCH_SIZE
            self.test_set = self.data_reader.test_set[:length].reshape((-1, BATCH_SIZE)).to(DEVICE)
        self.train_set = None

    def init_participant(self, global_model: TargetModel, participant_index):
        """
        Initialize the current model as a participant
        :return: None
        """
        self.participant_index = participant_index
        self.load_parameters(global_model.get_flatten_parameters())
        self.load_data()

    def share_gradient(self, noise_scale=NOISE_SCALE, agr=False):
        """
        Participants share gradient to the aggregator
        :return: None
        """
        gradient = self.get_epoch_gradient()
        gradient, indices = select_by_threshold(gradient, GRADIENT_EXCHANGE_RATE, GRADIENT_SAMPLE_THRESHOLD)
        noise = torch.randn(gradient.size()).to(DEVICE)
        noise = (noise / noise.norm()) * noise_scale * gradient.norm()
        # print("gradient norm before add noise {}".format(gradient.norm()), end = "")
        gradient += noise
        # print("gradient norm after add noise {}".format(gradient.norm()))
        if agr:
            self.aggregator.agr_loss_gradient_collect(gradient, indices)
        else:
            self.aggregator.collect(gradient, indices=indices, source=self.participant_index)
        return gradient

    def apply_gradient(self):
        """
        Global model applies the gradient
        :return: None
        """
        parameters = self.get_flatten_parameters().to(DEVICE)
        parameters += self.aggregator.get_outcome(reset=True).to(DEVICE)
        self.load_parameters(parameters)

    def collect_parameters(self, parameter: torch.Tensor):
        """
        Participants collect parameters from the global model
        :param parameter: the parameters shared by the global model
        :return: None
        """
        to_load = self.get_flatten_parameters().to(DEVICE)
        parameter, indices = select_by_threshold(parameter, PARAMETER_EXCHANGE_RATE, PARAMETER_SAMPLE_THRESHOLD)
        to_load[indices] = parameter[indices]
        self.load_parameters(to_load)

    def check_member(self, dataindex, participant_ind):
        result = None
        if dataindex in self.data_reader.get_train_set(participant_ind):
            result = True
        # print(dataindex,result)
        return result

    def detect_node_side(self, member, last_batch):
        out_list = []
        ground_truth = []
        correct_set = []
        # member_x,member_y = self.data_reader.get_batch(member)
        print("participant 0")
        print(member)
        for batch in member:
            # print(len(self.data_reader.get_train_set()))
            for i in batch:
                print(i)
                sample_x, sample_y = self.data_reader.get_batch(i.type(torch.int64))
                out = self.model(sample_x)
                prediction = torch.max(out, -1).indices.to(DEVICE)
                probs = torch.nn.functional.softmax(out, dim=0).to(DEVICE)
                if prediction == sample_y and i.item() not in correct_set:
                    correct_set.append(i.item())
                print(prediction, sample_y)
                # print(sample_y)
                # print(out, out[sample_y])
                out_list.append([float(probs[sample_y]), int(prediction), int(sample_y), i.item()])
        for i in last_batch:
            # print(i)
            # print(self.data_reader.train_set_last_batch)
            sample_x, sample_y = self.data_reader.get_batch(i.type(torch.int64))
            out = self.model(sample_x)
            prediction = torch.max(out, -1).indices.to(DEVICE)
            probs = torch.nn.functional.softmax(out, dim=0).to(DEVICE)
            if prediction == sample_y and i.item() not in correct_set:
                correct_set.append(i.item())
            # print(prediction)
            # print(sample_y)
            # print(out, out[sample_y])
            out_list.append([float(probs[sample_y]), int(prediction), int(sample_y), i.item()])
        return out_list, correct_set

    def detect_node_side_vector(self, member, last_batch):
        correct_set_dic = {}
        correct_set = []
        out_list = []
        ground_truth = []
        # member_x,member_y = self.data_reader.get_batch(member)
        for batch in member:
            # print(len(self.data_reader.get_train_set()))
            for i in batch:
                sample_x, sample_y = self.data_reader.get_batch(i.type(torch.int64))
                out = self.model(sample_x)
                prediction = torch.max(out, -1).indices.to(DEVICE)
                probs = torch.nn.functional.softmax(out, dim=0).to(DEVICE)
                if prediction == sample_y and i.item() not in correct_set_dic.keys():
                    correct_set_dic[i.item()] = float(probs[sample_y])
                    correct_set.append(i.item())
                # print(prediction)
                # print(sample_y)
                # print(out, out[sample_y])
                out_list.append([float(probs[sample_y]), int(prediction), int(sample_y), i.item()])
        for i in last_batch:
            # print(i)
            # print(self.data_reader.train_set_last_batch)
            sample_x, sample_y = self.data_reader.get_batch(i.type(torch.int64))
            out = self.model(sample_x)
            prediction = torch.max(out, -1).indices.to(DEVICE)
            probs = torch.nn.functional.softmax(out, dim=0).to(DEVICE)
            if prediction == sample_y and i.item() not in correct_set_dic.keys():
                correct_set_dic[i.item()] = float(probs[sample_y])
                correct_set.append(i.item())

            # print(prediction)
            # print(sample_y)
            # print(out, out[sample_y])
            out_list.append([float(probs[sample_y]), int(prediction), int(sample_y), i.item()])
        return out_list, correct_set, correct_set_dic

    def check_member_label(self, member):
        attacker_ground = []
        pred_label = {}
        label_flag = []
        out_list = []
        for i in member:
            sample_x, sample_y = self.data_reader.get_batch(i.type(torch.int64))
            out = self.model(sample_x)
            prediction = torch.max(out, -1).indices.to(DEVICE)
            probs = torch.nn.functional.softmax(out, dim=0).to(DEVICE)
            # print(prediction)
            # print(sample_y)
            # print(out, out[sample_y])
            out_list.append([float(probs[sample_y]), int(prediction), int(sample_y), i.item()])

        return pred_label, attacker_ground, label_flag, out_list

    def check_label_on_samples(self, participant_ind, attacker_samples):
        # result,membernode_list,nonmember_list = self.data_reader.get_black_box_batch()
        # attack_samples, members, non_members = reader.get_black_box_batch()
        # print(self.member_list)
        # member_list = []
        # attack_samples, members, nonmembers = self.get_attacker_sample()
        print(attacker_samples)
        for sample_ind in range(len(attacker_samples)):
            # print(sample_ind.numpy())
            # print(self.data_reader.train_set[participant_ind])
            # print(self.attack_samples[16])
            if attacker_samples[sample_ind] in self.data_reader.train_set and sample_ind not in [x[0] for x in
                                                                                                 self.member_list]:
                # print(sample_y)
                self.member_list.append((sample_ind, attacker_samples[sample_ind]))
        print(self.member_list)
        attack_x, attack_y = self.data_reader.get_batch(attacker_samples.type(torch.int64)).to(DEVICE)
        sample_x, sample_y = self.data_reader.get_batch(
            self.data_reader.train_set[participant_ind].type(torch.int64)).to(DEVICE)
        # print(sample_y[])
        attacker_ground = []
        pred_label = {}
        label_flag = []
        out_list = []
        for i in self.member_list:
            # print(i[1].item())
            out = self.model(attack_x[i[0]])
            prediction = torch.max(out, -1).indices.to(DEVICE)
            probs = torch.nn.functional.softmax(out, dim=0).to(DEVICE)
            print(prediction)
            attack_label = attack_y[i[0]]
            print(attack_label)
            print(out, out[attack_label])
            out_list.append([float(probs[attack_label]), int(prediction), int(attack_label), i[1].item()])
            if prediction == attack_label:
                label_flag.append("same {}, ground {}".format(i, int(attack_label)))
            else:
                label_flag.append(
                    "different {}, ground {}, predicted label {} ".format(i, int(attack_label), int(prediction)))
            pred_label[i[1]] = int(prediction)
            # print(i[1],attack_label)
            attacker_ground.append((int(i[1]), int(attack_label)))

        # pred_label = prediction[sample]
        #     print("the ith sample is {}, predicted label is {}".format(i,prediction))
        # print(pred_label)
        return pred_label, attacker_ground, label_flag, out_list

    def check_nonmember_sample(self, participant_ind, attacker_samples):
        # for i,sample_ind in enumerate(self.data_reader.train_set[participant_ind]):
        #     # if sample_ind in self.data_reader.train_set[5]:
        #         # print(sample_ind)
        #     if sample_ind not in [x[1] for x in self.member_list]:
        #         self.nonmember_list.append((i,int(sample_ind)))
        nonmembers_x, nonmembers_y = self.data_reader.get_batch(attacker_samples[2:].type(torch.int64))
        # print(nonmembers_y)
        # sample_x,sample_y = self.data_reader.get_batch(self.data_reader.train_set)
        # print(sample_y[])
        nonmember_ground = []
        pred_label_nonmember = {}
        for i in range(len(attacker_samples[2:])):
            # print(i)
            out = self.model(nonmembers_x[i]).to(DEVICE)
            prediction = torch.max(out, -1).indices.to(DEVICE)
            nonmember_label = nonmembers_y[i]
            # print(i[0])
            # print(prediction)
            # print(attack_label)
            pred_label_nonmember[int(attacker_samples[2:][i])] = int(prediction)
            nonmember_ground.append((int(attacker_samples[2:][i]), int(nonmember_label)))
        # print(pred_label_nonmember)
        # pred_label = prediction[sample]
        #     print("the ith sample is {}, predicted label is {}".format(i,prediction))

        return pred_label_nonmember, nonmember_ground

    def detect_attack(self, participant_ind):
        print("start detecting attack")
        targeted_samples_monitor = []
        print(self.train_set)
        print(self.last_train_batch)
        for batch_num, batch in enumerate(self.train_set):
            # print("in batch {}".format(batch_num))
            # print(self.data_reader.get_train_set(0))
            for num, i in enumerate(batch):
                sample_x, sample_y = self.data_reader.get_batch(i.type(torch.int64))
                out = self.model(sample_x)
                prediction = torch.max(out, -1).indices.to(DEVICE)
                probs = torch.nn.functional.softmax(out, dim=0).to(DEVICE)
                print(prediction, sample_y)
                if prediction != sample_y:
                    targeted_samples_monitor.append(i)
                print("monitor_list", targeted_samples_monitor)
        if self.last_train_batch != None:
            for num, i in enumerate(self.last_train_batch):
                sample_x, sample_y = self.data_reader.get_batch(i.type(torch.int64))
                out = self.model(sample_x)
                prediction = torch.max(out, -1).indices.to(DEVICE)
                probs = torch.nn.functional.softmax(out, dim=0).to(DEVICE)
                print(prediction, sample_y)
                if prediction != sample_y:
                    targeted_samples_monitor.append(i)
                print("monitor_list", targeted_samples_monitor)
                # if len(targeted_samples_monitor) == 3:
                # print(targeted_samples_monitor)
        return targeted_samples_monitor

    def detect_attack_vector(self, correct_set, correct_set_dic):
        targeted_samples_monitor = []
        for batch_num, batch in enumerate(self.train_set):
            # print(self.data_reader.get_train_set(0))
            for num, i in enumerate(batch):
                sample_x, sample_y = self.data_reader.get_batch(i.type(torch.int64))
                out = self.model(sample_x)
                prediction = torch.max(out, -1).indices.to(DEVICE)
                probs = torch.nn.functional.softmax(out, dim=0).to(DEVICE)
                # print(prediction,sample_y)
                print(correct_set)
                print(correct_set_dic)
                if i.item() in correct_set and correct_set_dic[i.item()] - float(probs[sample_y]) > 0.2:
                    targeted_samples_monitor.append(i)
                elif i.item() in correct_set and prediction != sample_y:
                    targeted_samples_monitor.append(i)

                # print(targeted_samples_monitor)

                # if len(targeted_samples_monitor) == 3:
                # print(targeted_samples_monitor)
        return targeted_samples_monitor

    def nomarl_detection(self, participant_ind):
        out_list = []
        correct_set = []
        for batch_num, batch in enumerate(self.data_reader.get_train_set(participant_ind)):
            # print(self.data_reader.get_train_set(0))
            for num, i in enumerate(batch):
                sample_x, sample_y = self.data_reader.get_batch(i.type(torch.int64))
                out = self.model(sample_x)
                prediction = torch.max(out, -1).indices.to(DEVICE)
                probs = torch.nn.functional.softmax(out, dim=0).to(DEVICE)
                # print(i.item())
                # print(prediction,sample_y)
                if prediction == sample_y and i.item() not in correct_set:
                    correct_set.append(i.item())
                # print(prediction)
                # print(sample_y)
                # print(out, out[sample_y])
                out_list.append([float(probs[sample_y]), int(prediction), int(sample_y), i.item()])

                # print(targeted_samples_monitor)

                # if len(targeted_samples_monitor) == 3:
                # print(targeted_samples_monitor)
        return out_list, correct_set

    def nomarl_detection_vector(self, participant_ind):
        out_list = []
        correct_set_dic = {}
        correct_set = []
        for batch_num, batch in enumerate(self.data_reader.get_train_set(participant_ind)):
            # print(self.data_reader.get_train_set(0))
            for num, i in enumerate(batch):
                sample_x, sample_y = self.data_reader.get_batch(i.type(torch.int64))
                out = self.model(sample_x)
                prediction = torch.max(out, -1).indices.to(DEVICE)
                probs = torch.nn.functional.softmax(out, dim=0).to(DEVICE)
                # print(prediction,sample_y)
                if prediction == sample_y and i.item() not in correct_set_dic.keys():
                    correct_set_dic[i.item()] = float(probs[sample_y])
                    correct_set.append(i.item())
                # print(prediction)
                # print(sample_y)
                # print(out, out[sample_y])
                out_list.append([float(probs[sample_y]), int(prediction), int(sample_y), i.item()])

                # print(targeted_samples_monitor)

                # if len(targeted_samples_monitor) == 3:
                # print(targeted_samples_monitor)
        return out_list, correct_set, correct_set_dic

    def del_defence(self, position):
        # print(len(self.data_reader.train_set))
        self.train_set, self.last_train_batch = self.data_reader.del_samples(position, self.train_set,
                                                                             self.last_train_batch)
        # return remove_number
        # print(len(self.data_reader.train_set[participant_index]))


class WhiteBoxMalicious(FederatedModel):
    """
    Representing the malicious participant trying to collect data for a white-box membership inference attack
    #TODO
    """

    def __init__(self, reader: DataReader, aggregator: Aggregator):
        super(WhiteBoxMalicious, self).__init__(reader, aggregator, 0)
        self.members = None
        self.non_members = None
        self.batch_x = None
        self.batch_y = None
        self.rest = None
        # try:
        #     if SAMPLE_TYPE == "single":
        #         self.attack_samples, self.members, self.non_members = reader.get_black_box_batch_fixed_single()
        #         self.batch_x, self.batch_y = self.data_reader.get_batch(self.attack_samples[:3])
        #         self.rest = self.attack_samples[3:]
        # except NameError:
        #     self.attack_samples, self.members, self.non_members = reader.get_black_box_batch_fixed()
        # self.attack_samples, self.members, self.non_members = reader.get_black_box_batch()
        try:
            if DEFAULT_AGR == FANG or DEFAULT_AGR == FL_TRUST:
                # if DEFAULT_AGR == FANG:
                self.attack_samples, self.members, self.non_members = reader.get_black_box_batch_fixed_balance_class()
            else:
                self.attack_samples, self.members, self.non_members = reader.get_black_box_batch_fixed()

                # print(len(self.members),len(self.non_members))
        except NameError:
            self.attack_samples, self.members, self.non_members = reader.get_black_box_batch_fixed()
            # print(len(self.members), len(self.non_members))
        # self.batch_x, self.batch_y = self.data_reader.get_batch(self.members)
        self.rest = self.non_members
        # self.all_samples = self.get_attack_sample()
        # try:
        #     if DEFAULT_AGR == FANG or FL_TRUST:
        #         self.attack_samples_fixed, self.members_fixed, self.non_members_fixed = reader.get_black_box_batch_fixed_balance_class()
        # except NameError:
        #     self.attack_samples_fixed, self.members_fixed, self.non_members_fixed = reader.get_black_box_batch_fixed()
        # self.attack_samples = self.all_samples

        self.descending_samples = None
        self.shuffled_labels = {}
        self.shuffle_labels()

        self.global_gradient = torch.zeros(self.get_flatten_parameters().size())
        self.last_round_shared_grad = None
        self.pred_history = []
        self.pred_history.append([])
        self.pred_history.append([])
        self.pred_history_new = {}
        self.confidence_history = []
        self.confidence_history.append([])
        self.confidence_history.append([])
        self.member_prediction = None
        self.member_intersections = {}
        self.sample_hist = {}

    def shuffle_label(self, ground_truth):
        result = ground_truth[torch.randperm(ground_truth.size()[0])]
        for i in range(ground_truth.size()[0]):
            while result[i].eq(ground_truth[i]):
                result[i] = torch.randint(ground_truth.max(), (1,))
        return result

    def record_targeted_samples(self):
        for member in self.members:
            members_x, members_y = self.data_reader.get_batch(member.type(torch.int64))
            out = self.model(members_x)
            prediction = torch.max(out, -1).indices
            if prediction == members_y:
                result = 1
            else:
                result = 0
            self.sample_hist[int(member)] = int(self.sample_hist.get(int(member), 0)) + result

    def chose_targeted_samples(self):
        result = []
        rest = []
        sorted_sample_hist = sorted(self.sample_hist.items(), key=lambda x: x[1], reverse=True)
        print(sorted_sample_hist)
        count = 0
        for key, value in sorted_sample_hist:
            count += 1
            if count >= 5:
                rest.append(torch.tensor(key).to(DEVICE))
            else:
                result.append(torch.tensor(key).to(DEVICE))
        print(result)
        self.batch_x, self.batch_y = self.data_reader.get_batch([torch.tensor(result).to(DEVICE)].type(torch.int64))
        self.rest = torch.cat([torch.tensor(rest).to(DEVICE), self.non_members])
        return result

    def attacker_sample(self):
        # self.batch_x, self.batch_y = self.data_reader.get_batch(self.attack_samples_fixed[:3])
        # self.shuffled_y = self.shuffle_label(self.batch_y)
        # self.batch_x, self.batch_y = self.data_reader.get_batch(self.members)
        return self.attack_samples, self.members, self.non_members

    def get_samples(self, members):
        # self.attack_samples, self.members, self.non_members = attack_samples,members,non_members
        self.members = members
        self.batch_x, self.batch_y = self.data_reader.get_batch(self.members.type(torch.int64))
        print(self.members)

    def target_participants(self, participant_index):
        self.attack_samples_fixed = self.attack_samples_fixed[:int(
            (NUMBER_OF_ATTACK_SAMPLES * BLACK_BOX_MEMBER_RATE / NUMBER_OF_PARTICIPANTS) * (participant_index + 1))]

    def optimized_gradient_ascent(self, batch_size=BATCH_SIZE, ascent_factor=ASCENT_FACTOR,
                                  mislead=False, mislead_factor=1, cover_factor=2):
        """
        Take one step of gradient ascent, the returned gradient is a combination of ascending gradient, descending
        gradient, and misleading gradient
        :return: gradient generated
        """
        print("ascent_factor {}, cover factor {}".format(ascent_factor, cover_factor))
        # attack_x, attack_y = self.data_reader.get_batch(self.attack_samples_fixed[:3])
        cache = self.get_flatten_parameters()
        self.load_parameters(cache)
        out = self.model(self.batch_x)
        print(len(out))
        # print(self.attack_samples_fixed[:int((NUMBER_OF_ATTACK_SAMPLES*BLACK_BOX_MEMBER_RATE/NUMBER_OF_PARTICIPANTS)*(participant+1))][0],out[0],attack_y[0])
        loss = self.loss_function(out, self.batch_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        gradient = self.get_flatten_parameters() - cache
        ascent_gradient = - ascent_factor * gradient
        # to_load = cache + gradient
        # self.load_parameters(to_load)
        # self.load_parameters(cache)
        if RESERVED_SAMPLE != 0:
            self.load_parameters(cache)
            cover_samples = self.data_reader.reserve_set
            i = 0
            while i * batch_size < len(cover_samples):
                batch_index = cover_samples[i * batch_size:(i + 1)]
                x, y = self.data_reader.get_batch(batch_index.type(torch.int64))
                out = self.model(x)
                loss = self.loss_function(out, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                i += 1
            cov_gradient = cover_factor * self.get_flatten_parameters() - cache
            # gradient, indices = select_by_threshold(cov_gradient,GRADIENT_EXCHANGE_RATE, GRADIENT_SAMPLE_THRESHOLD)
            # self.aggregator.collect(gradient, indices)
        # print(self.attack_samples_fixed[int((NUMBER_OF_ATTACK_SAMPLES*BLACK_BOX_MEMBER_RATE/NUMBER_OF_PARTICIPANTS)*(participant+1)):],len(self.attack_samples_fixed[int((NUMBER_OF_ATTACK_SAMPLES*BLACK_BOX_MEMBER_RATE/NUMBER_OF_PARTICIPANTS)*(participant+1)):]))
        #     new_load = cache + gradient - cov_gradient

        self.load_parameters(cache)
        x_rest, y_rest = self.data_reader.get_batch(self.rest.type(torch.int64))
        out = self.model(x_rest)
        loss = self.loss_function(out, y_rest)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        normal_gradient = self.get_flatten_parameters() - cache
        if RESERVED_SAMPLE != 0:
            final_gradient = cov_gradient + normal_gradient + ascent_gradient
        else:
            final_gradient = cover_factor * normal_gradient + ascent_gradient
            # final_gradient = normal_gradient
        gradient, indices = select_by_threshold(final_gradient, GRADIENT_EXCHANGE_RATE, GRADIENT_SAMPLE_THRESHOLD)
        self.aggregator.collect(gradient, indices)

        return gradient

    def optimized_descent(self, ascent_factor=ASCENT_FACTOR, cover_factor=0, batch_size=BATCH_SIZE):
        try:
            cover_factor = COVER_FACTOR
            mislead_factor = MISLEAD_FACTOR
        except NameError:
            cover_factor = 0
            mislead_factor = 0

        cache = self.get_flatten_parameters()
        attack_gradient = self.optimized_train(attack=True, mislead_factor=mislead_factor)
        cover_samples = self.data_reader.reserve_set
        i = 0
        while i * batch_size < len(cover_samples):
            batch_index = cover_samples[i * batch_size:(i + 1) * batch_size]
            x, y = self.data_reader.get_batch(batch_index.type(torch.int64))
            out = self.model(x)
            loss = self.loss_function(out, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            i += 1
        cov_gradient = self.get_flatten_parameters() - cache

        return cov_gradient - attack_gradient

    def prune_data(self, label=KEEP_CLASS):
        if label == None:
            return None
        batch_x, batch_y = self.data_reader.get_batch(self.attack_samples.type(torch.int64))
        index = []
        for i in range(len(batch_y)):
            if batch_y[i] == label:
                # self.attack_samples=torch.cat([self.attack_samples[0:i],self.attack_samples[i+1:]])
                index.append(i)

        self.attack_samples = self.attack_samples[index].to(DEVICE)

    def shuffle_labels(self, iteration=WHITE_BOX_SHUFFLE_COPIES):
        """
        Shuffle the labels in several random permutation, to be used as misleading labels
        it will repeat the given iteration times denote as k, k different copies will be saved
        """
        max_label = torch.max(self.data_reader.labels).item()
        for i in range(iteration):
            shuffled = self.data_reader.labels[torch.randperm(len(self.data_reader.labels))]
            for j in torch.nonzero(shuffled == self.data_reader.labels):
                shuffled[j] = (shuffled[j] + torch.randint(max_label, [1]).item()) % max_label
            self.shuffled_labels[i] = shuffled

    def gradient_ascent(self, ascent_factor=ASCENT_FACTOR, batch_size=BATCH_SIZE,
                        adaptive_factor=FRACTION_OF_ASCENDING_SAMPLES, mislead=False, mislead_factor=1):
        """
        Take one step of gradient ascent, the returned gradient is a combination of ascending gradient, descending
        gradient, and misleading gradient
        :return: gradient generated
        """
        cache = self.get_flatten_parameters()
        threshold = round(len(self.attack_samples) * adaptive_factor)

        # Perform gradient ascent for ascending samples
        if RESERVED_SAMPLE != 0:
            cover_samples = self.data_reader.reserve_set
            i = 0
            while i * batch_size < len(cover_samples):
                batch_index = cover_samples[i * batch_size:(i + 1) * batch_size]
                x, y = self.data_reader.get_batch(batch_index.type(torch.int64))
                out = self.model(x)
                loss = self.loss_function(out, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                i += 1
            cov_gradient = self.get_flatten_parameters() - cache
            self.load_parameters(cache)
        # Perform gradient descent for the rest of samples
        i = 0
        while i * batch_size < len(self.attack_samples):
            batch_index = self.attack_samples[i * batch_size:(i + 1) * batch_size]
            x, y = self.data_reader.get_batch(batch_index.type(torch.int64))
            out = self.model(x)
            loss = self.loss_function(out, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            i += 1
        desc_gradient = self.get_flatten_parameters() - cache
        if RESERVED_SAMPLE != 0:
            final_gradient = desc_gradient + cov_gradient
        else:
            final_gradient = desc_gradient
        return final_gradient

    def train(self, mislead_factor=3, norm_scalling=1, attack=False, mislead=True,
              ascent_factor=ASCENT_FACTOR, ascent_fraction=FRACTION_OF_ASCENDING_SAMPLES, white_box_optimize=False):
        """
        Start a white-box training
        """
        self.record_targeted_samples()
        # try:
        #     mislead_factor =MISLEAD_FACTOR
        # except NameError:
        #     mislead = 1
        #
        # try:
        norm_scalling = NORM_SCALLING
        # except NameError:
        #     norm_scalling = 1
        print("train")
        gradient = self.gradient_ascent(ascent_factor=ascent_factor, adaptive_factor=ascent_fraction, mislead=mislead,
                                        mislead_factor=mislead_factor)
        gradient, indices = select_by_threshold(gradient, GRADIENT_EXCHANGE_RATE, GRADIENT_SAMPLE_THRESHOLD)
        if attack:
            random_key = torch.randint(1, 10, [1]).item()
            if self.global_gradient is not None and white_box_optimize and random_key < 8:
                norm = self.global_gradient.norm()
                gradient += self.global_gradient
                gradient = gradient * norm * norm_scalling / gradient.norm()
            self.last_round_shared_grad = gradient
            # self.aggregator.collect(gradient, indices)
        self.aggregator.collect(gradient, indices)
        return gradient

    def collect_parameters(self, parameter: torch.Tensor):
        """
        Save the parameters from last round before collect new parameters
        """
        cache = self.get_flatten_parameters()
        super(WhiteBoxMalicious, self).collect_parameters(parameter)
        self.global_gradient = self.get_flatten_parameters() - cache

    def examine_sample_gradients(self, monitor_window=512):
        """
        Examine the gradients for each sample, used to compare
        """
        cache = self.get_flatten_parameters()
        monitor = {}
        for i in range(len(self.attack_samples)):
            print("\r Evaluating monitor window for attack sample {}/{}".format(i, len(self.attack_samples)), end="")
            sample = self.attack_samples[i]
            x = self.data_reader.data[sample]
            y = self.data_reader.labels[sample]
            x = torch.vstack([x] * BATCH_SIZE)
            y = torch.hstack([y] * BATCH_SIZE)
            out = self.model(x)
            loss = self.loss_function(out, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            gradient = self.get_flatten_parameters() - cache
            sorted = torch.sort(gradient, descending=True)
            monitor[i] = sorted.indices[:monitor_window], \
                         sorted.values[:monitor_window]
            self.load_parameters(cache)
        print(" Monitor window generated.")
        return monitor

    def evaluate_member_accuracy(self):
        """
        Evaluate the accuracy rate of members in the attack samples
        """
        result = []
        batch_x, batch_y = self.data_reader.get_batch(self.members.type(torch.int64))
        with torch.no_grad():
            out = self.model(batch_x)
        prediction = torch.max(out, 1).indices.to(DEVICE)
        # accurate = torch.sum(prediction == batch_y).to(DEVICE)
        accurate = (prediction == batch_y).sum().to(DEVICE)
        for i in range(len(self.members)):
            out = self.model(batch_x[i])
            prediction = torch.max(out, -1).indices
            # print(prediction)
            flag = (prediction == batch_y[i])
            result.append((self.members[i], batch_y[i]))
        # acc = torch.tensor(accurate/ len(batch_y))
        return accurate / len(batch_y), result

    def evaluate_non_member_accuracy(self):
        """
        Evaluate the accuracy rate of non-members in the attack samples
        """
        batch_x, batch_y = self.data_reader.get_batch(self.non_members.type(torch.int64))
        with torch.no_grad():
            out = self.model(batch_x)
        prediction = torch.max(out, 1).indices.to(DEVICE)
        accurate = (prediction == batch_y).sum().to(DEVICE)
        return accurate / len(batch_y)

    def optimized_evaluate_member_accuracy(self):
        """
        Evaluate the accuracy rate of members in the attack samples
        """
        selected_participants = RobustMechanism.appearence_list
        union = []
        for i in selected_participants:
            if i != 5:
                union.append(self.member_intersections[i])
            else:
                pass

        effective_members = torch.unique(torch.cat(union, 0).to(DEVICE))
        if len(effective_members) > 0:
            batch_x, batch_y = self.data_reader.get_batch(effective_members.type(torch.int64))
            with torch.no_grad():
                out = self.model(batch_x).to(DEVICE)
            prediction = torch.max(out, 1).indices
            accurate = (prediction == batch_y).sum()
            return accurate / len(batch_y)
        return 0

    def optimized_evaluate_non_member_accuracy(self):
        """
        Evaluate the accuracy rate of non-members in the attack samples
        """
        selected_participants = RobustMechanism.appearence_list
        to_union = []
        for i in range(NUMBER_OF_PARTICIPANTS):
            if i not in selected_participants:
                if i != 5:
                    to_union.append(i)
                else:
                    pass
        union = [self.non_members.to(DEVICE)]
        for i in to_union:
            union.append(self.member_intersections[i])

        effective_non_members = torch.unique(torch.cat(union, 0).to(DEVICE))
        if len(effective_non_members) > 0:
            batch_x, batch_y = self.data_reader.get_batch(effective_non_members.type(torch.int64))
            with torch.no_grad():
                out = self.model(batch_x)
            prediction = torch.max(out, 1).indices.to(DEVICE)
            accurate = (prediction == batch_y).sum()
            return accurate / len(batch_y)
        return 0

    def optimized_evaluation_init(self):
        """
        Calculate the intersection of self.members and the train set of each participant
        """
        for i in range(NUMBER_OF_PARTICIPANTS):
            self.member_intersections[i] = \
                torch.tensor(
                    np.intersect1d(self.data_reader.get_train_set(i).to(DEVICE), self.attack_samples.to(DEVICE)))

    def get_pred_member(self, rounds):
        pred_member = {}
        batch_x, batch_y = self.data_reader.get_batch(self.attack_samples.type(torch.int64))
        for i in range(len(self.member_prediction)):
            if self.pred_history_new[rounds][i] == 0:
                continue
            for j in range(NUMBER_OF_PARTICIPANTS):
                if self.attack_samples[i] in self.data_reader.get_train_set(j):
                    pred_member[i] = [batch_y[i], j]
                    break
                pred_member[i] = [batch_y[i], "No"]
        return pred_member

    def partial_gradient_ascent(self, ascent_factor=ASCENT_FACTOR, agr=False, ascent_fraction=0.5):
        """
        Perform gradient ascent on only a subset of the attack samples
        """
        cache = self.get_flatten_parameters()
        rand_perm = torch.randperm(len(self.attack_samples))
        ascent_count = round(len(self.attack_samples) * ascent_fraction)
        batch_x, batch_y = self.data_reader.get_batch(self.attack_samples[rand_perm[:ascent_count]].type(torch.int64))
        out = self.model(batch_x)
        loss = self.loss_function(out, batch_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        gradient = self.get_flatten_parameters() - cache
        gradient = - ascent_factor * gradient
        to_load = cache + gradient
        self.load_parameters(to_load)
        batch_x, batch_y = self.data_reader.get_batch(self.attack_samples[rand_perm[ascent_count:]].type(torch.int64))
        out = self.model(batch_x)
        loss = self.loss_function(out, batch_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        gradient = self.get_flatten_parameters() - cache
        gradient, indices = select_by_threshold(gradient, GRADIENT_EXCHANGE_RATE, GRADIENT_SAMPLE_THRESHOLD)
        if agr:
            self.aggregator.agr_loss_gradient_collect(gradient, indices)
        else:
            self.aggregator.collect(gradient, indices)

    def evaluate_attack_result(self, adaptive_prediction=True, adaptive_strategy=SCORE_BASED_STRATEGY, rounds=None):
        """
        Evaluate the attack result, return the overall accuracy, member accuracy, and precise
        """
        true_member = 0
        false_member = 0
        true_non_member = 0
        false_non_member = 0
        attack_result = []
        ground_truth = []
        batch_x, batch_y = self.data_reader.get_batch(self.members.type(torch.int64))
        out = self.model(batch_x)
        prediction = torch.max(out, 1).indices
        confidence = torch.max(out, 1).values
        accuracy = ((prediction == batch_y).sum() / len(self.members)).to(DEVICE)
        low_confidence_samples = torch.sort(confidence).indices[:torch.round(len(self.members) * (1 - accuracy)).int()]
        classification_counter = 0
        if adaptive_prediction and self.last_round_shared_grad is not None and self.global_gradient.norm() != 0:
            monitor = self.examine_sample_gradients()
            if adaptive_strategy == NORM_BASED_STRATEGY:
                norm_diff = self.get_classification_norm_based(monitor)
        for i in range(len(self.members)):
            if adaptive_prediction and self.last_round_shared_grad is not None and self.global_gradient.norm() != 0:
                if adaptive_strategy == SCORE_BASED_STRATEGY:
                    membership = self.get_classification_score_based(monitor[i][0])
                elif adaptive_strategy == NORM_BASED_STRATEGY:
                    membership = i not in norm_diff[:torch.round(len(self.members) * (1 - accuracy))]
                if i in low_confidence_samples and prediction[i] == batch_y[i]:
                    classification_counter += 1
                    if membership:
                        attack_result.append(1)
                    else:
                        attack_result.append(0)
                else:
                    if prediction[i] == batch_y[i]:
                        attack_result.append(1)
                    else:
                        attack_result.append(0)

            else:
                if prediction[i] == batch_y[i]:
                    attack_result.append(1)
                else:
                    attack_result.append(0)
                # print(attack_result)
            if self.members[i] in self.data_reader.train_set:
                ground_truth.append(1)
            else:
                ground_truth.append(0)
            if (attack_result[i] == 1) and (ground_truth[i] == 1):
                true_member += 1
            elif (attack_result[i] == 1) and (ground_truth[i] == 0):
                false_member += 1
            elif (attack_result[i] == 0) and (ground_truth[i] == 0):
                true_non_member += 1
            else:
                false_non_member += 1
        if rounds is not None:
            self.pred_history_new[rounds] = attack_result
        print("overall {} classified attacks".format(classification_counter))
        self.member_prediction = attack_result
        return true_member, false_member, true_non_member, false_non_member

    def evaluate_label_attack_result(self, base_pred=TRAIN_EPOCH):
        """
        Evaluate the attack result, return the overall accuracy, and precise
        """
        true_member = 0
        false_member = 0
        true_non_member = 0
        false_non_member = 0
        attack_result = []
        ground_truth = []
        batch_x, batch_y = self.data_reader.get_batch(self.attack_samples.type(torch.int64))
        out = self.model(batch_x)
        prediction = torch.max(out, 1).indices

        for i in range(len(self.attack_samples)):
            # Defalut: use the pred before attack as base

            if prediction[i] == self.pred_history[0][base_pred][i]:
                attack_result.append(1)
            else:
                attack_result.append(0)
            if self.attack_samples[i] in self.data_reader.train_set:
                ground_truth.append(1)
            else:
                ground_truth.append(0)

            if (attack_result[i] == 1) and (ground_truth[i] == 1):
                true_member += 1
            elif (attack_result[i] == 1) and (ground_truth[i] == 0):
                false_member += 1
            elif (attack_result[i] == 0) and (ground_truth[i] == 0):
                true_non_member += 1
            else:
                false_non_member += 1

        return true_member, false_member, true_non_member, false_non_member

    def in_round(self, dataindex, filte_client=False):
        result = False
        owner = 5
        samples = self.attack_samples
        for j in self.aggregator.robust.appearence_list:
            if filte_client:
                samples = self.current_attack_samples
            if samples[dataindex] in self.data_reader.get_train_set(j) and j != 5:
                result = True
                owner = j
                return result
            else:
                pass
        return result

    def evaluate_optimized_attack_result(self, adaptive_prediction=True, adaptive_strategy=SCORE_BASED_STRATEGY):
        """
        Evaluate the attack result, return the overall accuracy, member accuracy, and precise
        """
        print(self.aggregator.robust.appearence_list)
        if 5 in self.aggregator.robust.appearence_list:
            in_number = len(self.aggregator.robust.appearence_list) - 1
        else:
            in_number = 5
        true_member = 0
        false_member = 0
        true_non_member = 0
        false_non_member = 0
        nonmember_count = 0
        member0 = 0
        member1 = 0
        member2 = 0
        member3 = 0
        member4 = 0
        attack_result = []
        ground_truth = []
        attack_member_sample = []
        member_list = []
        # for i in [0,1,2,3]:
        #     batch_x1,batch_y = self.data_reader
        for i in range(len(self.attack_samples)):
            if self.in_round(i):
                attack_member_sample.append(self.attack_samples[i])
                member_list.append(i)
        print(len(attack_member_sample))
        attack_member_sample = torch.tensor(attack_member_sample).to(DEVICE)
        current_attack_samples = torch.cat([attack_member_sample, self.non_members[
            torch.randperm(round((in_number / NUMBER_OF_PARTICIPANTS) * len(self.attack_samples) / 2))]]).to(DEVICE)
        current_attack_samples = current_attack_samples[torch.randperm(len(current_attack_samples))].to(DEVICE)
        self.current_attack_samples = current_attack_samples
        batch_x, batch_y = self.data_reader.get_batch(current_attack_samples.type(torch.int64))

        out = self.model(batch_x)
        prediction = torch.max(out, 1).indices
        confidence = torch.max(out, 1).values
        accuracy = ((prediction == batch_y).sum() / len(current_attack_samples))
        low_confidence_samples = torch.sort(confidence).indices[
                                 :torch.round(len(self.attack_samples) * (1 - accuracy)).int()]
        classification_counter = 0
        if adaptive_prediction and self.last_round_shared_grad is not None and self.global_gradient.norm() != 0:
            monitor = self.examine_sample_gradients()
            if adaptive_strategy == NORM_BASED_STRATEGY:
                norm_diff = self.get_classification_norm_based(monitor)

        for i in range(len(current_attack_samples)):
            # result, owner = self.in_round(i)
            # member_list.append(owner)
            # if owner == 0 :
            #     member0 += 1
            # elif owner == 1 :
            #     member1 += 1
            # elif owner == 2 :
            #     member2 += 1
            # elif owner == 3 :
            #     member3 += 1
            # elif owner == 4 :
            #     member4 += 1
            # if result or self.attack_samples[i] not in self.data_reader.train_set:
            if prediction[i] == batch_y[i]:
                # print(prediction[i])
                attack_result.append(1)
                # print(attack_result)
            elif prediction[i] != batch_y[i]:
                attack_result.append(0)
            if self.in_round(i, filte_client=True):
                ground_truth.append(1)
            elif current_attack_samples[i] not in self.data_reader.train_set:
                nonmember_count += 1
                ground_truth.append(0)
            else:
                ground_truth.append(3)

            # print(attack_result)
            # print(ground_truth)
            # print(i)
            if (attack_result[i] == 1) and (ground_truth[i] == 1):
                true_member += 1
            elif (attack_result[i] == 1) and (ground_truth[i] == 0):
                false_member += 1
            elif (attack_result[i] == 0) and (ground_truth[i] == 0):
                true_non_member += 1
            elif (attack_result[i] == 0) and (ground_truth[i] == 1):
                false_non_member += 1
            else:
                pass
        # print("member 0 has {}, member 1 has {}, member 2 has {}, member 3 has {}, member 4 has {}".format(member0,member1,member2, member3, member4))
        # print("overall {} classified attacks".format(classification_counter))
        # print("all samples belongs to {}".format(member_list))
        return true_member, false_member, true_non_member, false_non_member

    def get_classification_score_based(self, monitor_window):
        """
        Get the prediction outcome of a given monitor window using the membership score
        """
        # strong_attack = (self.last_round_shared_grad < 0).sum() > (len(monitor_window)*accuracy)
        member_score = torch.logical_and((self.last_round_shared_grad[monitor_window] < 0),
                                         (self.global_gradient[monitor_window] >= 0)).sum()
        non_member_score = torch.logical_and((self.last_round_shared_grad[monitor_window] < 0),
                                             (self.global_gradient[monitor_window] < 0)).sum()
        # print("member score = {}, non_member score = {}".format(member_score, non_member_score))
        return member_score >= non_member_score

    def get_classification_norm_based(self, monitor: dict):
        """
        Get the prediction outcome of a given monitor using norm of the difference
        """
        all_diff = torch.zeros(self.attack_samples.size())
        for key in monitor.keys():
            index, value = monitor[key]
            norm1 = (value - self.global_gradient[index]).norm()
            norm2 = (value - self.last_round_shared_grad[index]).norm()
            all_diff[key] = norm2 - norm1
        all_diff = torch.sort(all_diff).indices
        return all_diff
