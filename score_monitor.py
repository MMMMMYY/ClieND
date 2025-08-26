import torch
import numpy as np
import os
import sys
import logging
import time
import scipy.fftpack as fft
import pandas as pd
import collections


def get_net_rankings(model):
    """
    Get the ranking of scores for each layer
    :param model: 
    :return: 
    """
    local_ranked_update = collections.defaultdict(list)
    for i, v in model.named_modules():
        if hasattr(v, "popup_scores"):
            if getattr(v, "popup_scores") is not None:
                _, rank = v.popup_scores.detach().flatten().sort()
                local_ranked_update[str(i)] = rank[None, :]
    return local_ranked_update

def sanity_check_paramter_updates(model, last_ckpt):
    """
        Check whether weigths/popup_scores gets updated or not compared to last ckpt.
        ONLY does it for 1 layer (to avoid computational overhead)
    """
    for i, v in model.named_modules():
        score_changed_list = []
        if hasattr(v, "weight") and hasattr(v, "popup_scores"):
            if getattr(v, "weight") is not None:
                w1 = getattr(v, "weight").data.cpu()
                w2 = last_ckpt[i + ".weight"].data.cpu()
            if getattr(v, "popup_scores") is not None:
                s1 = getattr(v, "popup_scores").data.cpu()
                s2 = last_ckpt[i + ".popup_scores"].data.cpu()
            return not torch.allclose(w1, w2), not torch.allclose(s1, s2)

def check_scores_ranking_changes(model, last_ckpt):
    """
    Check whether the ranking of scores changes or not
    :param model:
    :param last_ckpt:
    :return:
    """
    rank_chaged_list = []
    for i, v in model.named_modules():
        if hasattr(v, "popup_scores"):
            if getattr(v, "popup_scores") is not None:
                s1 = getattr(v, "popup_scores").data.cpu()
                s2 = last_ckpt[i + ".popup_scores"].data.cpu()
                rank1 = torch.argsort(s1)
                rank2 = torch.argsort(s2)
                rank_chaged_list.append(rank1 != rank2)
    return rank_chaged_list

def count_changed_rank(model, last_ckpt):
    """
    Count the number of changed scores
    :param model:
    :param last_ckpt:
    :return:
    """
    total_changed = 0
    for i, v in model.named_modules():
        if hasattr(v, "popup_scores"):
            if getattr(v, "popup_scores") is not None:
                s1 = getattr(v, "popup_scores").data.cpu()
                s2 = last_ckpt[i + ".popup_scores"].data.cpu()
                rank_changed = torch.argsort(s1) != torch.argsort(s2)
                total_changed += torch.sum(rank_changed)
    return total_changed

def get_rank_of_difference_scores(model, last_ckpt):
    """
    Get the rank of the difference of scores
    :param model:
    :param last_ckpt:
    :return:
    """
    rank_diff_list = []
    for i, v in model.named_modules():
        if hasattr(v, "popup_scores"):
            if getattr(v, "popup_scores") is not None:
                s1 = getattr(v, "popup_scores").data.cpu()
                s2 = last_ckpt[i + ".popup_scores"].data.cpu()
                rank_diff = torch.argsort(s1 - s2)
                rank_diff_list.append(rank_diff)
    return rank_diff_list

def report_score_changed_neurons(model, last_ckpt):
    """
    Report the largest score changed for 10 neurons
    :param model:
    :param last_ckpt:
    :return:
    """
    first_10_neurons = []
    model_score_diff = []
    layer_score_diff = []
    for i, v in model.named_modules():
        if hasattr(v, "popup_scores"):
            if getattr(v, "popup_scores") is not None:
                s1 = getattr(v, "popup_scores").data.cpu()
                s2 = last_ckpt[i + ".popup_scores"].data.cpu()
                model_score_diff.append(s1.numpy() - s2.numpy())
    # print(s1)
    return perform_dct2d(model_score_diff)

def perform_dct2d(score_diff):
    """
    Perform DCT on the score_diff
    :param score_diff:
    :return:
    """
    for i in range(len(score_diff)):
        score_diff[i] = fft.dct(fft.dct(score_diff[i], axis=0, norm='ortho'), axis=1, norm='ortho')
        # print(score_diff[i].shape)
    return score_diff

def get_score_sum(model):
    '''
    Get the sum of scores for each layer
    :param model:
    :return:

    '''
    local_sum = 0
    for i, v in model.named_modules():
        if hasattr(v, "popup_scores"):
            if getattr(v, "popup_scores") is not None:
                local_sum += torch.sum(v.popup_scores).data.cpu().detach().numpy()
    return local_sum

def get_score_sum_diff(last_sum, current_sum):
    """
    Get the difference of sum of scores
    :param last_sum:
    :param current_sum:
    :return:
    """
    return current_sum - last_sum

def get_topk_neurons(model, k=0.1):
    """
    Get the top k neurons index with the largest score
    :param model:
    :param k:
    :return: the index of top k neurons
    """
    topk_neurons = []
    for i, v in model.named_modules():
        if hasattr(v, "popup_scores"):
            if getattr(v, "popup_scores") is not None:
                _, rank = v.popup_scores.flatten().sort()
                # get the index of top k neurons in adj
                indices = rank[-int(k * len(rank)):]
                topk_neurons.append(indices.data.cpu().detach().numpy())
    # print(topk_neurons)

    return topk_neurons

def get_score_sum_topk(model, topk_neurons):
    """
    Get the sum of scores for top k neurons
    :param model:
    :param topk_neurons:
    :return:
    """
    local_sum = 0
    for i, array in enumerate(topk_neurons):
        name, module = list(model.named_modules())[i]  # Get module by index
        if hasattr(module, "popup_scores") and module.popup_scores is not None:
            score_array = module.popup_scores.flatten()
            local_sum += score_array[i].data.cpu().detach().numpy()
                        # local_sum += torch.sum(module.popup_scores[idx]).data.cpu().detach().numpy()

    return local_sum

def get_the_threshold(model):
    """
    Get the threshold for top k neurons
    :param model:
    :param k:
    :return:
    """
    min = 0
    max = 0
    for i, v in model.named_modules():
        if hasattr(v, "popup_scores"):
            if getattr(v, "popup_scores") is not None:
                min = torch.min(v.popup_scores).data.cpu().detach().numpy()
                max = torch.max(v.popup_scores).data.cpu().detach().numpy()
    return min, max

def get_the_threshold_topk(model, topk_neurons):
    """
    Get the threshold for top k neurons
    :param model:
    :param k:
    :return:
    """
    min = 0
    max = 0
    for i, array in enumerate(topk_neurons):
        name, module = list(model.named_modules())[i]  # Get module by index
        if hasattr(module, "popup_scores") and module.popup_scores is not None:
            min = torch.min(module.popup_scores).data.cpu().detach().numpy()
            max = torch.max(module.popup_scores).data.cpu().detach().numpy()
    return min, max


def check_neuron_score(model, min, max):
    """
    Check whether the score of neurons are beyond the range, and return the sum of neurons
    :param model:
    :param threshold:
    :return:
    """
    local_sum = 0
    for i, v in model.named_modules():
        if hasattr(v, "popup_scores"):
            if getattr(v, "popup_scores") is not None:
                s = v.popup_scores.data.cpu().detach().numpy()
                local_sum += np.sum(s < min) + np.sum(s > max)
    return local_sum

def check_neuron_score_topk(model, topk_neurons, min, max):
    """
    Check whether the score of neurons are beyond the range, and return the sum of neurons
    :param model:
    :param threshold:
    :return:
    """
    local_sum = 0
    for i, array in enumerate(topk_neurons):
        name, module = list(model.named_modules())[i]  # Get module by index
        if hasattr(module, "popup_scores") and module.popup_scores is not None:
            s = module.popup_scores.data.cpu().detach().numpy()
            local_sum += np.sum(s < min) + np.sum(s > max)
    return local_sum

def return_score(model):
    scores = []
    max_rows = 0
    for i, v in model.named_modules():
        if hasattr(v, "popup_scores"):
            if getattr(v, "popup_scores") is not None:
                s1 = getattr(v, "popup_scores").data.cpu()
                scores.append(s1)
                s1 = s1.view(1, -1) # Reshape the scores tensor
                # print("the shape of s1 is: ", s1.shape)
                # if i not in score_hist:
                #     score_hist[i] = s1
                # else:
                #     score_hist[i] = torch.cat((score_hist[i], s1), 0).to(device)
                max_rows = max(max_rows, s1.shape[0])
    # print(score_hist)
    # Check the shapes of the arrays in scores
    # shapes = [score.shape for score in scores]
    # if len(set(shapes)) > 1:
    #     print("Warning: Arrays in 'scores' have mismatched dimensions:", shapes)
    # Flatten and concatenate all scores tensors
    resized_scores = torch.cat([score.view(-1) for score in scores])

    # Reshape the concatenated scores tensor
    concatenated_scores = resized_scores.view(-1, len(resized_scores))
    print("the shape of concatenated_scores is: ", concatenated_scores.shape)
    scores_list = concatenated_scores[0].tolist()

    reshape_scores = np.array(scores_list).reshape(1, -1)
    return reshape_scores

def calculate_round_difference(round1_scores, round2_scores):
    if round1_scores is not None and round2_scores is not None:
        # Flatten the scores to ensure 1D arrays
        round1 = round1_scores.flatten()
        round2 = round2_scores.flatten()

        # Ensure the scores have the same shape
        if round1.shape != round2.shape:
            raise ValueError("Scores from both rounds must have the same shape.")

        # Calculate the element-wise differences
        differences = round1 - round2

        # Sort the differences in descending order
        sorted_differences = np.sort(differences)[::-1]
        return sorted_differences
    else:
        print("Invalid input: Scores for one or both rounds are missing.")
        return None

def sort_scores(reshape_scores):
    if reshape_scores is not None:
        # Flatten and sort the scores from high to low
        sorted_scores = np.sort(reshape_scores.flatten())[::-1]
        return sorted_scores
    else:
        print("Invalid input: No scores to sort.")
        return None

def return_weight(model):
    weights = []
    max_rows = 0
    for i, v in model.named_modules():
        if hasattr(v, "weight"):
            if getattr(v, "weight") is not None:
                s1 = getattr(v, "weight").data.cpu()
                weights.append(s1)
                max_rows = max(max_rows, s1.shape[0])

    # Check the shapes of the arrays in scores
    # shapes = [score.shape for score in scores]
    # if len(set(shapes)) > 1:
    #     print("Warning: Arrays in 'scores' have mismatched dimensions:", shapes)
    # Flatten and concatenate all scores tensors
    resized_weights = torch.cat([weight.view(-1) for weight in weights])

    # Reshape the concatenated scores tensor
    concatenated_weights = resized_weights.view(-1, len(resized_weights))
    # print("the shape of concatenated_weights is: ", concatenated_weights.shape)
    weights_list = concatenated_weights[0].tolist()

    reshape_weights = np.array(weights_list).reshape(1, -1)
    return reshape_weights
