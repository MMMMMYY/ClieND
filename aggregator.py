import numpy as np
from cifar10_util import *
from typing import List, Any
import torch
import pandas as pd
import hdbscan
import copy
from models import *
# from constants import *
# from data_reader import DataReader
# from common import DEVICE

# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from constants import *
from utils.crowdguard import *
from torch.utils.data import DataLoader, TensorDataset



class Aggregator:
    """
    The aggregator class collecting gradients calculated by participants and plus together
    """

    def __init__(self, sample_gradients: torch.Tensor, robust_mechanism=None):
        """
        Initiate the aggregator according to the tensor size of a given sample
        :param sample_gradients: The tensor size of a sample gradient
        :param robust_mechanism: the robust mechanism used to aggregate the gradients
        """
        self.sample_gradients = sample_gradients.to(DEVICE)
        self.collected_gradients = []
        self.counter = 0
        self.counter_by_indices = torch.ones(self.sample_gradients.size()).to(DEVICE)
        self.robust = RobustMechanism(robust_mechanism)

        # AGR related parameters
        self.agr_model = None #Global model Fang required

    def reset(self):
        """
        Reset the aggregator to 0 before next round of aggregation
        """
        self.collected_gradients = []
        self.counter = 0
        self.counter_by_indices = torch.ones(self.sample_gradients.size())
        self.agr_model_calculated = False

    def collect(self, gradient: torch.Tensor,source, indices=None, sample_count=None):
        """
        Collect one set of gradients from a participant
        :param gradient: The gradient calculated by a participant
        :param souce: The source of the gradient, used for AGR verification
        :param indices: the indices of the gradient, used for AGR verification

        """
        if sample_count is None:
            self.collected_gradients.append(gradient)
            if indices is not None:
                self.counter_by_indices[indices] += 1
            self.counter += 1
        else:
            self.collected_gradients.append(gradient * sample_count)
            if indices is not None:
                self.counter_by_indices[indices] += sample_count
            self.counter += sample_count

    def get_outcome(self, reset=False, by_indices=False):
        """
        Get the aggregated gradients and reset the aggregator if needed, apply robust aggregator mechanism if needed
        :param reset: Whether to reset the aggregator after getting the outcome
        :param by_indices: Whether to aggregate by indices
        """
        if by_indices:
            result = sum(self.collected_gradients) / self.counter_by_indices
        else:
            result = self.robust.getter(self.collected_gradients, malicious_user=NUMBER_OF_ADVERSARY)
        if reset:
            self.reset()
        return result

    def agr_model_acquire(self, model):
        """
        Make use of the given model for AGR verification
        :param model: The model used for AGR verification
        """
        self.agr_model = model
        self.robust.agr_model_acquire(model)


class RobustMechanism:
    """
    The robust aggregator applied in the aggregator
    """
    appearence_list = [0,1,2,3,4]
    status_list = []

    def __init__(self, robust_mechanism):
        self.type = robust_mechanism
        if robust_mechanism is None:
            self.function = self.naive_average
        elif robust_mechanism == TRMEAN:
            self.function = self.trmean
        elif robust_mechanism in (KRUM, MULTI_KRUM):
            self.function = self.multi_krum
        elif robust_mechanism == BULYAN:
            self.function = self.bulyan
        elif robust_mechanism == MEDIAN:
            self.function = self.median
        elif robust_mechanism == FANG:
            self.function = self.Fang_defense
        elif robust_mechanism == FL_TRUST:
            self.function = self.fl_trust
        self.agr_model = None


    def agr_model_acquire(self, model):
        """
        Acquire the model used for LRR and ERR verification in Fang Defense
        The model must have the same parameters as the global model
        """
        self.agr_model = model

    def naive_average(self, input_gradients, malicious_user:int):
        """
        The naive aggregator
        """
        return torch.mean(input_gradients, 0)

    def trmean(self, input_gradients, malicious_user: int):
        """
        The trimmed mean
        """
        # sorted_results = torch.sort(input_gradients, 0)[1]
        sorted_updates,sorted_results  = torch.sort(input_gradients, 0)
        print(sorted_results)
        if malicious_user*2 < len(input_gradients):
            print(sorted_results[malicious_user: -malicious_user])
            return torch.mean(sorted_updates[malicious_user: -malicious_user], 0)
        else:
            return torch.mean(sorted_updates, 0)

    def median(self, input_gradients, malicious_user: int):
        """
        The median AGR
        """
        median,index = torch.median(input_gradients, 0)
        print(input_gradients)
        # print(len(index))
        # print("median index",index.cpu().numpy().tolist().count(30))
        return median

    def multi_krum(self, all_updates, n_attackers):
        """
        The multi-krum method copied from Mengyap's update
        """
        multi_k =  (self.type == MULTI_KRUM)
        candidates = []
        candidate_indices = []
        remaining_updates = all_updates
        all_indices = np.arange(len(all_updates))

        while len(remaining_updates) > 2 * n_attackers + 2:
            # torch.cuda.empty_cache()
            distances = []
            for update in remaining_updates:
                distance = []
                for update_ in remaining_updates:
                    distance.append(torch.norm((update - update_)) ** 2)
                distance = torch.Tensor(distance).float()
                distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)

            distances = torch.sort(distances, dim=1)[0]
            scores = torch.sum(distances[:, :len(remaining_updates) - 2 - n_attackers], dim=1)
            indices = torch.argsort(scores)[:len(remaining_updates) - 2 - n_attackers]

            candidate_indices.append(all_indices[indices[0]])
            all_indices = np.delete(all_indices, indices[0])
            candidates = remaining_updates[indices[0]][None, :] if not len(candidates) else torch.cat(
                (candidates, remaining_updates[indices[0]][None, :]), 0)
            remaining_updates = torch.cat((remaining_updates[:indices[0]], remaining_updates[indices[0] + 1:]), 0)
            if not multi_k:
                break

        # aggregate = torch.mean(candidates, dim=0)
        print("Selected candicates = {}".format(np.array(candidate_indices)))
        # if not multi_k:
        #     RobustMechanism.appearence_list = candidate_indices
        RobustMechanism.appearence_list = candidate_indices
        return torch.mean(candidates, dim=0)

    def bulyan(self, all_updates, n_attackers):
        """
        The code for robust AGR Bulyan
        """
        nusers = all_updates.shape[0]
        bulyan_cluster = []
        candidate_indices = []
        remaining_updates = all_updates
        all_indices = np.arange(len(all_updates))

        while len(bulyan_cluster) < (nusers - 2 * n_attackers):
            # torch.cuda.empty_cache()
            distances = []
            for update in remaining_updates:
                distance = []
                for update_ in remaining_updates:
                    distance.append(torch.norm((update - update_)) ** 2)
                distance = torch.Tensor(distance).float()
                distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)
            # print(distances)

            distances = torch.sort(distances, dim=1)[0]

            scores = torch.sum(distances[:, :len(remaining_updates) - 2 - n_attackers], dim=1)
            indices = torch.argsort(scores)[:len(remaining_updates) - 2 - n_attackers]
            if not len(indices):
                break
            candidate_indices.append(all_indices[indices[0]])
            all_indices = np.delete(all_indices, indices[0])
            bulyan_cluster = remaining_updates[indices[0]][None, :] if not len(bulyan_cluster) else torch.cat(
                (bulyan_cluster, remaining_updates[indices[0]][None, :]), 0)
            remaining_updates = torch.cat((remaining_updates[:indices[0]], remaining_updates[indices[0] + 1:]), 0)

        # print('dim of bulyan cluster ', bulyan_cluster.shape)

        n, d = bulyan_cluster.shape
        param_med = torch.median(bulyan_cluster, dim=0)[0]
        sort_idx = torch.argsort(torch.abs(bulyan_cluster - param_med), dim=0)
        sorted_params = bulyan_cluster[sort_idx, torch.arange(d)[None, :]]
        print("Selected participants = {}".format(np.array(candidate_indices)))
        return torch.mean(sorted_params[:n - 2 * n_attackers], dim=0)

    def Fang_defense(self, input_gradients: torch.Tensor, malicious_user: int):
        """
        The LRR mechanism proposed in Fang defense
        """
        # Get the baseline loss and accuracy without removing any of the inputs
        all_avg = torch.mean(input_gradients, 0)
        base_loss, base_acc = self.agr_model.test_gradients(all_avg)
        loss_impact = []
        err_impact = []
        # Get all the loss value and accuracy without ith input
        RobustMechanism.status_list = []
        for i in range(len(input_gradients)):
            avg_without_i = (sum(input_gradients[:i]) + sum(input_gradients[i+1:])) / (input_gradients.size(0) - 1)
            ith_loss, ith_acc = self.agr_model.test_gradients(avg_without_i)
            loss_impact.append(torch.tensor(base_loss - ith_loss))
            err_impact.append(torch.tensor(ith_acc - base_acc))
            RobustMechanism.status_list.append((i,ith_acc,ith_loss))
        loss_impact = torch.hstack(loss_impact)
        err_impact = torch.hstack(err_impact)
        loss_rank = torch.argsort(loss_impact, dim=-1)
        acc_rank = torch.argsort(err_impact, dim=-1)
        result = []
        # print(loss_rank, acc_rank)
        for i in range(len(input_gradients)):
            if i in loss_rank[:-malicious_user] and i in acc_rank[:-malicious_user]:
                result.append(i)
        # print("Selected inputs are from participants number {}".format(result))
        RobustMechanism.appearence_list = result
        return torch.mean(input_gradients[result], dim=0)

    def PCA_defense(self, input_gradients: torch.Tensor, malicious_user: int):
        """
        The proposed PCA based defense
        """
        all_mean = torch.mean(input_gradients, dim=0)
        input_gradients -= all_mean
        #todo

    def getter(self, x, malicious_user=NUMBER_OF_ADVERSARY):
        """
        The getter method applying the robust AGR
        """
        x = torch.vstack(x)
        return self.function(x, malicious_user)

    def fl_trust(self,input_gradients,malicious_user):
        replica = input_gradients.clone()
        grad_zero = self.agr_model.get_gradzero()
        print(grad_zero)
        grad_zero = grad_zero.unsqueeze(0)
        # grad_zero = torch.tensor(list(grad_zero)*5)
        cos = torch.nn.CosineSimilarity(eps = 1e-5)
        relu = torch.nn.ReLU()
        norm = grad_zero.norm()
        scores = []
        # print(input_gradients)
        # print(input_gradients.squeeze(0))
        for i in input_gradients:
            i = i.unsqueeze(0)
            score = cos(i, grad_zero)
            scores.append(score)
        # score = cos(input_gradients.squeeze(0), grad_zero)
        # print(score)
        scores = torch.tensor([item for item in scores]).to(DEVICE)
        print(scores)
        scores = relu(scores)

        grad = torch.nn.functional.normalize(replica)* norm
        grad =(grad.transpose(0, 1)*scores).transpose(0,1)
        grad =torch.sum(grad, dim=0)/scores.sum()
        return grad


def fl_trust(input_gradients):
    replica = input_gradients.clone()
    grad_zero = self.agr_model.get_gradzero()
    print(grad_zero)
    grad_zero = grad_zero.unsqueeze(0)
    # grad_zero = torch.tensor(list(grad_zero)*5)
    cos = torch.nn.CosineSimilarity(eps = 1e-5)
    relu = torch.nn.ReLU()
    norm = grad_zero.norm()
    scores = []
    # print(input_gradients)
    # print(input_gradients.squeeze(0))
    for i in input_gradients:
        i = i.unsqueeze(0)
        score = cos(i, grad_zero)
        scores.append(score)
    # score = cos(input_gradients.squeeze(0), grad_zero)
    # print(score)
    scores = torch.tensor([item for item in scores]).to(DEVICE)
    print(scores)
    scores = relu(scores)

    grad = torch.nn.functional.normalize(replica)* norm
    grad =(grad.transpose(0, 1)*scores).transpose(0,1)
    grad =torch.sum(grad, dim=0)/scores.sum()
    return grad


def flame(fed_model, input_gradients):
    weight_accumulator = {}
    for name, params in fed_model.state_dict().items():
        weight_accumulator[name] = torch.zeros_like(params)

    clients_weight_ = []
    clients_weight_total = []
    for data in input_gradients:
        client_weight = torch.tensor([])
        client_weight_total = torch.tensor([])

        for name, params in data.items():
            client_weight = torch.cat((client_weight, params.reshape(-1).cpu()))
            if '.weight' in name or 'bias' in name:
                client_weight_total = torch.cat(
                    (client_weight_total, (params + fed_model.state_dict()[name]).reshape(-1).cpu()))

        clients_weight_.append(client_weight)
        clients_weight_total.append(client_weight_total)

    clients_weight_ = torch.stack(clients_weight_)
    clients_weight_total = torch.stack(clients_weight_total)
    num_clients = clients_weight_total.shape[0]
    euclidean = (clients_weight_ ** 2).sum(1).sqrt()
    med = euclidean.median()

    clients_weight_total = clients_weight_total.double()
    cluster = hdbscan.HDBSCAN(metric="cosine", algorithm="generic", min_cluster_size=num_clients // 2 + 1,
                              min_samples=1, allow_single_cluster=True)

    cluster.fit(clients_weight_total)
    predict_good = []
    predict_bad = []
    for i, j in enumerate(cluster.labels_):
        if j == 0:
            predict_good.append(i)
        else:
            predict_bad.append(i)

    print(cluster.labels_)
    print("Good clients recognized:", predict_good)
    print("Bad clients recognized:", predict_bad)

    # Print malicious clients
    print("Malicious clients recognized:", np.where(cluster.labels_ != 0)[0])

    for i, data in enumerate(input_gradients):
        gama = med.div(euclidean[i])
        if gama > 1:
            gama = 1

        for name, params in data.items():
            params.data = (params.data * gama).to(params.data.dtype)

    num_in = 0
    for i, data in enumerate(input_gradients):
        if cluster.labels_[i] == 0:
            num_in += 1
            for name, params in data.items():
                weight_accumulator[name].add_(params)

    model_aggregate(fed_model, weight_accumulator, num_in)

    lamda = 0.000012
    for name, param in fed_model.named_parameters():
        if 'bias' in name or 'bn' in name:
            continue
        std = lamda * med * param.data.std()
        noise = torch.normal(0, std, size=param.size()).cuda()
        param.data.add_(noise)


def model_aggregate(fed_model,weight_accumulator, num):
    for name, data in fed_model.state_dict().items():

        update_per_layer = weight_accumulator[name] / num

        if data.type() != update_per_layer.type():
            data.add_(update_per_layer.to(torch.int64))
        else:
            data.add_(update_per_layer)

def multi_krum(all_updates, n_attackers):
    """
    The multi-krum method copied from Mengyap's update
    """
    multi_k =  True
    candidates = []
    candidate_indices = []
    remaining_updates = all_updates
    all_indices = np.arange(len(all_updates))

    while len(remaining_updates) > 2 * n_attackers + 2:
        # torch.cuda.empty_cache()
        distances = []
        for update in remaining_updates:
            distance = []
            for update_ in remaining_updates:
                distance.append(torch.norm((update - update_)) ** 2)
            distance = torch.Tensor(distance).float()
            distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)

        distances = torch.sort(distances, dim=1)[0]
        scores = torch.sum(distances[:, :len(remaining_updates) - 2 - n_attackers], dim=1)
        indices = torch.argsort(scores)[:len(remaining_updates) - 2 - n_attackers]

        candidate_indices.append(all_indices[indices[0]])
        all_indices = np.delete(all_indices, indices[0])
        candidates = remaining_updates[indices[0]][None, :] if not len(candidates) else torch.cat(
            (candidates, remaining_updates[indices[0]][None, :]), 0)
        remaining_updates = torch.cat((remaining_updates[:indices[0]], remaining_updates[indices[0] + 1:]), 0)
        if not multi_k:
            break

    # aggregate = torch.mean(candidates, dim=0)
    print("Selected candicates = {}".format(np.array(candidate_indices)))
    # if not multi_k:
    #     RobustMechanism.appearence_list = candidate_indices
    return torch.mean(candidates, dim=0)

def crowdguard(client_models, global_model, data, label, device):

    # Validate models locally and gather votes
    # Create a dictionary of models from client weights
    # Apply gradients to the fed_model
    def apply_gradients(model, grads):
        model.zero_grad()
        grad_tensors = torch.cat([grad.view(-1) for grad in grads])
        start_idx = 0
        for param in model.parameters():
            end_idx = start_idx + param.numel()
            if param.grad is None:
                param.grad = torch.zeros_like(param.data)
            param.grad.data.copy_(grad_tensors[start_idx:end_idx].view_as(param))
            start_idx = end_idx
        return model

    # Create a dictionary of models from client grads
    # Function to copy model state without deepcopy

    client_model, _ = return_model(arch, 0.1, 0.9, parallel=False)
    # Create a dictionary of models from client grads
    all_models = {f'client_{i}': apply_gradients(client_model, client_models[i]) for i in
                  range(len(client_models))}

    all_votes = {}

    for i, model in enumerate(client_models):
        collaborator_name = f'client_{i}'
        # print(f"Validating model for collaborator {collaborator_name}")
        dataset = TensorDataset(data[i], label[i])
        local_data = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        votes = local_validation(collaborator_name, all_models, global_model, local_data, device)
        all_votes[collaborator_name] = votes

    # Aggregate votes
    benign_models = []
    for model_name, votes in all_votes.items():
        if votes[model_name] == VOTE_FOR_BENIGN:
            benign_models.append(all_models[model_name])

    if not benign_models:
        print("No benign models found. Using the original federated model.")
        return global_model

    # Average benign models to update the federated model
    state_dicts = [model.state_dict() for model in benign_models]
    fed_model_state_dict = global_model.state_dict()

    for key in fed_model_state_dict.keys():
        fed_model_state_dict[key] = torch.mean(torch.stack([state_dict[key].float() for state_dict in state_dicts]), dim=0)

    global_model.load_state_dict(fed_model_state_dict)

    return global_model

def local_validation(collaborator_name, all_models, global_model, local_data, device):
    # print(f"Performing model validation for collaborator {collaborator_name}")

    all_names = list(all_models.keys())
    all_model_list = [all_models[n] for n in all_names]
    own_client_index = all_names.index(collaborator_name)

    detected_suspicious_models = detect_suspicious_models(global_model, all_model_list, own_client_index,
                                                          local_data, device)
    votes_of_this_client = generate_votes(all_model_list, own_client_index, detected_suspicious_models, all_names)

    return {name: vote for name, vote in zip(all_names, votes_of_this_client)}

def detect_suspicious_models(global_model, all_models, own_client_index, local_data, device):
    return CrowdGuardClientValidation.validate_models(
        global_model, all_models, own_client_index, local_data, device
    )

def generate_votes(all_models, own_client_index, detected_suspicious_models, all_names):
    votes = []

    for c in range(len(all_models)):
        if c == own_client_index:
            votes.append(VOTE_FOR_BENIGN)
        elif c in detected_suspicious_models:
            votes.append(VOTE_FOR_POISONED)
        else:
            votes.append(VOTE_FOR_BENIGN)
    return votes


from sklearn.cluster import KMeans
import numpy as np


def cluster_client_gradients(user_grads, n_clusters=3):
    """
    Cluster all client gradients using KMeans.
    :param user_grads: A tensor of shape (num_clients, num_params) containing client gradients.
    :param n_clusters: The number of clusters to form.
    :return: A dictionary mapping cluster IDs to client indices.
    """
    user_grads_np = user_grads.cpu().numpy()  # Convert to numpy array for clustering

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(user_grads_np)

    # Group clients by cluster
    clusters = {i: [] for i in range(n_clusters)}
    for client_idx, cluster_id in enumerate(labels):
        clusters[cluster_id].append(client_idx)

    print("\nCluster Results:")
    for cluster_id, members in clusters.items():
        print(f"Cluster {cluster_id}: Clients {members}")

    return clusters

from sklearn.cluster import MeanShift

def cluster_client_gradients_mean_shift(user_grads):
    """
    Cluster client gradients using Mean Shift clustering.
    :param user_grads: A tensor of shape (num_clients, num_params) containing client gradients.
    :return: A dictionary mapping cluster IDs to client indices.
    """
    user_grads_np = user_grads.cpu().numpy()  # Convert to numpy array for clustering

    # Perform Mean Shift clustering
    mean_shift = MeanShift(bandwidth=None, bin_seeding=True)
    labels = mean_shift.fit_predict(user_grads_np)

    # Group clients by cluster
    clusters = {}
    for client_idx, cluster_id in enumerate(labels):
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(client_idx)

    print("\nMean Shift Cluster Results:")
    for cluster_id, members in clusters.items():
        print(f"Cluster {cluster_id}: Clients {members}")

    return clusters


def freqfed(grads, local_models, global_model):
    """
    FreqFed: Frequency-Domain Federated Learning.
    """
    grads_fft = torch.fft.rfft(grads, dim=1)

    n_freqs = grads_fft.shape[1]
    low_freq_cutoff = max(1, int(n_freqs * 0.2))

    low_freq_mag = torch.abs(grads_fft[:, :low_freq_cutoff])
    low_freq_cpu = low_freq_mag.cpu().numpy()

    if hdbscan is not None:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=max(2, grads.shape[0] // 3), min_samples=1)
        labels = clusterer.fit_predict(low_freq_cpu)
    else:
        kmeans = KMeans(n_clusters=2, random_state=0).fit(low_freq_cpu)
        labels = kmeans.labels_

    unique_labels, counts = np.unique(labels, return_counts=True)

    valid = unique_labels != -1
    if not valid.any(): return torch.mean(grads, dim=0)

    unique_labels = unique_labels[valid]
    counts = counts[valid]

    best_cluster_label = unique_labels[np.argmax(counts)]
    benign_indices = np.where(labels == best_cluster_label)[0]

    return torch.mean(grads[benign_indices], dim=0)


def sherpa(grads, local_models, global_model, n_attacker):
    """
    SHERPA: Robust Aggregation using clustering.
    """
    n_clients = grads.shape[0]
    n_components = min(n_clients, 5)

    mean_grad = torch.mean(grads, dim=0)
    centered_grads = grads - mean_grad

    try:
        U, S, V = torch.pca_lowrank(centered_grads, q=n_components)
        reduced_features = torch.matmul(centered_grads, V[:, :n_components]).cpu().numpy()
    except:
        reduced_features = grads.cpu().numpy()

    kmeans = KMeans(n_clusters=2, random_state=42).fit(reduced_features)
    labels = kmeans.labels_

    counts = np.bincount(labels)
    benign_label = np.argmax(counts)

    benign_indices = np.where(labels == benign_label)[0]

    return torch.mean(grads[benign_indices], dim=0)


def flshield(local_models, global_model, val_data, val_label, device):
    """
    FLShield: Validation-based aggregation.
    """
    if val_data is None or val_label is None:
        return torch.mean(torch.stack([torch.cat([p.data.view(-1) for p in m.parameters()]) for m in local_models]),
                          dim=0)

    criterion = nn.CrossEntropyLoss()
    losses = []

    with torch.no_grad():
        for model in local_models:
            model.eval()
            output = model(val_data.to(device))
            loss = criterion(output, val_label.to(device))
            losses.append(loss.item())

    losses = torch.tensor(losses)
    k = max(1, int(len(local_models) * 0.6))
    _, top_indices = torch.topk(losses, k, largest=False)

    selected_updates = []
    for i in top_indices:
        flat_params = torch.cat([p.data.view(-1) for p in local_models[i].parameters()])
        selected_updates.append(flat_params)

    return torch.mean(torch.stack(selected_updates), dim=0)


def feddefender(grads, local_models, global_model):
    """
    FedDefender: Sign-based defense.
    """
    avg_grad = torch.mean(grads, dim=0)
    signs = torch.sign(grads)
    sum_signs = torch.sum(signs, dim=0)

    consistency = torch.abs(sum_signs) / grads.shape[0]
    robust_grad = avg_grad * consistency

    return robust_grad