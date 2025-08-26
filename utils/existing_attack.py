import torch
import numpy as np
import pickle
import os
import torch.nn as nn


# label flipping attack
def get_label_flipping(inputs, targets):

    # print('Flipping labels')
    max_label = torch.max(targets).item()
    min_label = torch.min(targets).item()
    for i in range(len(targets)):
        # shuffled_label = torch.randint(min_label, max_label + 1, (1,)).item()
        # while shuffled_label == targets[i].item():
        #     shuffled_label = torch.randint(min_label, max_label + 1, (1,)).item()
        # targets[i] = shuffled_label
        if targets[i] == max_label:
            targets[i] = min_label
        elif targets[i] == min_label:
            targets[i] = max_label
    return inputs, targets


# FANG Trimean attack
def get_malicious_updates_fang_trmean(all_updates, deviation, n_attackers, epoch_num, compression='none', q_level=2,
                                      norm='inf'):
    b = 2
    max_vector = torch.max(all_updates, 0)[0]
    min_vector = torch.min(all_updates, 0)[0]

    max_ = (max_vector > 0).type(torch.FloatTensor).cuda()
    min_ = (min_vector < 0).type(torch.FloatTensor).cuda()

    max_[max_ == 1] = b
    max_[max_ == 0] = 1 / b
    min_[min_ == 1] = b
    min_[min_ == 0] = 1 / b

    max_range = torch.cat((max_vector[:, None], (max_vector * max_)[:, None]), dim=1)
    min_range = torch.cat(((min_vector * min_)[:, None], min_vector[:, None]), dim=1)

    rand = torch.from_numpy(np.random.uniform(0, 1, [len(deviation), n_attackers])).type(torch.FloatTensor).cuda()

    max_rand = torch.stack([max_range[:, 0]] * rand.shape[1]).T + rand * torch.stack(
        [max_range[:, 1] - max_range[:, 0]] * rand.shape[1]).T
    min_rand = torch.stack([min_range[:, 0]] * rand.shape[1]).T + rand * torch.stack(
        [min_range[:, 1] - min_range[:, 0]] * rand.shape[1]).T

    mal_vec = (torch.stack(
        [(deviation > 0).type(torch.FloatTensor)] * max_rand.shape[1]).T.cuda() * max_rand + torch.stack(
        [(deviation > 0).type(torch.FloatTensor)] * min_rand.shape[1]).T.cuda() * min_rand).T

    quant_mal_vec = []
    if compression != 'none':
        if epoch_num == 0: print('compressing malicious update')
        for i in range(mal_vec.shape[0]):
            mal_ = mal_vec[i]
            mal_quant = qsgd(mal_, s=q_level, norm=norm)
            quant_mal_vec = mal_quant[None, :] if not len(quant_mal_vec) else torch.cat(
                (quant_mal_vec, mal_quant[None, :]), 0)
    else:
        quant_mal_vec = mal_vec

    mal_updates = torch.cat((quant_mal_vec, all_updates), 0)

    return mal_updates


def lie_attack(all_updates, z):
    avg = torch.mean(all_updates, dim=0)
    std = torch.std(all_updates, dim=0)
    return avg + z * std


def our_attack_median(all_updates, model_re, n_attackers, dev_type='unit_vec'):
    if dev_type == 'unit_vec':
        deviation = model_re / torch.norm(model_re)  # unit vector, dir opp to good dir
    elif dev_type == 'sign':
        deviation = torch.sign(model_re)
    elif dev_type == 'std':
        deviation = torch.std(all_updates, 0)

    lamda = torch.Tensor([10.0]).cuda()  # compute_lambda_our(all_updates, model_re, n_attackers)

    threshold_diff = 1e-5
    prev_loss = -1
    lamda_fail = lamda
    lamda_succ = 0
    iters = 0
    while torch.abs(lamda_succ - lamda) > threshold_diff:
        mal_update = (model_re - lamda * deviation)
        mal_updates = torch.stack([mal_update] * n_attackers)
        mal_updates = torch.cat((mal_updates, all_updates), 0)

        agg_grads = torch.median(mal_updates, 0)[0]

        loss = torch.norm(agg_grads - model_re)

        if prev_loss < loss:
            lamda_succ = lamda
            lamda = lamda + lamda_fail / 2
        else:
            lamda = lamda - lamda_fail / 2

        lamda_fail = lamda_fail / 2
        prev_loss = loss

    mal_update = (model_re - lamda_succ * deviation)
    mal_updates = torch.stack([mal_update] * n_attackers)
    mal_updates = torch.cat((mal_updates, all_updates), 0)

    return mal_updates


def our_attack_score(all_updates, model_re, n_attackers, dev_type='unit_vec'):
    if dev_type == 'unit_vec':
        deviation = model_re / torch.norm(model_re)  # unit vector, dir opp to good dir
    elif dev_type == 'sign':
        deviation = torch.sign(model_re)
    elif dev_type == 'std':
        deviation = torch.std(all_updates, 0)

    lamda = torch.Tensor([10.0]).float().cuda()
    # print(lamda)
    threshold_diff = 1e-5
    lamda_fail = lamda
    lamda_succ = 0

    distances = []
    for update in all_updates:
        distance = torch.norm((all_updates - update), dim=1) ** 2
        distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)

    scores = torch.sum(distances, dim=1)
    min_score = torch.min(scores)
    del distances

    while torch.abs(lamda_succ - lamda) > threshold_diff:
        mal_update = (model_re - lamda * deviation)
        distance = torch.norm((all_updates - mal_update), dim=1) ** 2
        score = torch.sum(distance)

        if score <= min_score:
            # print('successful lamda is ', lamda)
            lamda_succ = lamda
            lamda = lamda + lamda_fail / 2
        else:
            lamda = lamda - lamda_fail / 2

        lamda_fail = lamda_fail / 2

    # print(lamda_succ)
    mal_update = (model_re - lamda_succ * deviation)

    return mal_update


def our_attack_dist(all_updates, model_re, n_attackers, dev_type='unit_vec'):
    if dev_type == 'unit_vec':
        deviation = model_re / torch.norm(model_re)  # unit vector, dir opp to good dir
    elif dev_type == 'sign':
        deviation = torch.sign(model_re)
    elif dev_type == 'std':
        deviation = torch.std(all_updates, 0)

    lamda = torch.Tensor([10.0]).float().cuda()
    # print(lamda)
    threshold_diff = 1e-5
    lamda_fail = lamda
    lamda_succ = 0

    distances = []
    for update in all_updates:
        distance = torch.norm((all_updates - update), dim=1) ** 2
        distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)

    max_distance = torch.max(distances)
    del distances

    while torch.abs(lamda_succ - lamda) > threshold_diff:
        mal_update = (model_re - lamda * deviation)
        distance = torch.norm((all_updates - mal_update), dim=1) ** 2
        max_d = torch.max(distance)

        if max_d <= max_distance:
            # print('successful lamda is ', lamda)
            lamda_succ = lamda
            lamda = lamda + lamda_fail / 2
        else:
            lamda = lamda - lamda_fail / 2

        lamda_fail = lamda_fail / 2

    mal_update = (model_re - lamda_succ * deviation)

    return mal_update


def rank_reverse(fed_model, cur_user_updates, n_attacker):
    for i in range(n_attacker):
        sum_args_sorts_mal = {}
        local_ranked_update = collections.defaultdict(list)
        for n, m in fed_model.named_modules():
            if hasattr(m, "popup_scores"):
                _, rank = m.popup_scores.detach().clone().flatten().sort()
                rank_arg = torch.sort(rank)[1]
                if str(n) in sum_args_sorts_mal:
                    sum_args_sorts_mal[str(n)] += rank_arg
                else:
                    sum_args_sorts_mal[str(n)] = rank_arg

                # reverse ranking
                rank_mal_agr = torch.sort(sum_args_sorts_mal[str(n)], descending=True)[1]
                local_ranked_update[str(n)] = rank_mal_agr[None, :]

        for n, m in fed_model.named_modules():
            if hasattr(m, "popup_scores"):
                if len(cur_user_updates[str(n)]) == 0:
                    cur_user_updates[str(n)] = local_ranked_update[str(n)][None, :]
                else:
                    cur_user_updates[str(n)] = torch.cat((cur_user_updates[str(n)],
                                                          local_ranked_update[str(n)][None, :]), 0)

    return cur_user_updates


def dropout_attack(fed_model, malicious_grads, n_attacker, k_percent):
    """
    Perform a dropout attack by masking top k percent of weights with malicious gradients.

    Args:
        fed_model (nn.Module): The federated model.
        malicious_grads (list of torch.Tensor): The list of malicious gradients for each parameter.
        n_attacker (int): The number of attackers.
        k_percent (float): The percentage of top weights to mask with malicious gradients.
    """
    # Ensure malicious_grads has the same structure as the model parameters
    assert len(malicious_grads) == len(
        list(fed_model.parameters())), "Mismatch in number of malicious gradients and model parameters"

    # Iterate through the model parameters and malicious gradients
    for param, malicious_grad in zip(fed_model.parameters(), malicious_grads):
        # Mask the top k percent of the current parameter's weights
        masked_param = mask_top_k_percent(param.data, k_percent)

        # Apply the malicious gradients to the masked weights
        masked_param += malicious_grad / n_attacker

        # Update the parameter with the new masked and modified weights
        param.data = masked_param


def trigger_single_image(image):
    """
    Adds a red square with a height/width of 6 pixels into
    the upper left corner of the given image.
    :param image tensor, containing the normalized pixel values of the image.
    The image will be modified in-place.
    :return given image
    """
    mean = torch.Tensor([0.485, 0.456, 0.406])
    std_dev = torch.Tensor([0.229, 0.224, 0.225])
    color = (torch.Tensor((1, 0, 0)) - mean) / std_dev
    image[:, 0:6, 0:6] = color.repeat((6, 6, 1)).permute(2, 1, 0)
    return image


def poison_data(samples_to_poison, labels_to_poison, pdr=0.5):
    """
    poisons a given local dataset, consisting of samples and labels, s.t.,
    the given ratio of this image consists of samples for the backdoor behavior
    :param samples_to_poison tensor containing all samples of the local dataset
    :labels_to_poison tensor containing all labels
    :return poisoned local dataset (samples, labels)
    """
    if pdr == 0:
        return samples_to_poison, labels_to_poison
    assert 0 < pdr <= 1.0
    samples_to_poison = samples_to_poison.clone()
    labels_to_poison = labels_to_poison.clone()

    dataset_size = samples_to_poison.shape[0]
    num_samples_to_poison = int(dataset_size * pdr)
    if num_samples_to_poison == 0:
        # corner case for tiny pdrs
        assert pdr > 0  # Already checked above
        assert dataset_size > 1
        num_samples_to_poison += 1

    indices = np.random.choice(dataset_size, size=num_samples_to_poison, replace=False)
    for image_index in indices:
        image = trigger_single_image(samples_to_poison[image_index])
        samples_to_poison[image_index] = image
    labels_to_poison[indices] = 2
    return samples_to_poison, labels_to_poison.long()

