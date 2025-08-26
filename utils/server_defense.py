import numpy as np
from cifar10_util import *
from typing import List, Any
import torch
import pandas as pd
# from constants import *
# from data_reader import DataReader
# from common import DEVICE

# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from constants import *

def FRL_Vote(model, user_updates, initial_scores):
    """
    This function is used to update the scores of the edges in the model
    :param FLmodel:
    :param user_updates:
    :param initial_scores:
    :return:
    """
    for n, m in model.named_modules():
        if hasattr(m, "popup_scores"):
            # sort by the edge index of this layer,
            # then retrieve the ranked index, i.e., score, of that layers after sort
            args_sorts = torch.sort(user_updates[str(n)])[1]
            sum_args_sorts = torch.sum(args_sorts, 0)
            idxx = torch.sort(sum_args_sorts)[1]
            # get back the scores of the previous model
            temp1 = m.popup_scores.detach().clone()
            temp1.flatten()[idxx] = initial_scores[str(n)]
            # re-assign back to the edge
            m.popup_scores = torch.nn.Parameter(temp1)
            del idxx, temp1



def fl_trust(model, loss, optimizer_fed, valdata, vallabel,input_gradients):
    # print(vallabel)
    replica = input_gradients.clone()
    #we use thw last user as the reference user
    grad_zero = input_gradients[-1]
    # grad_zero = get_gradzero(model, valdata, vallabel, loss, optimizer_fed)
    # print(grad_zero)
    grad_zero = grad_zero.unsqueeze(0)
    cos = torch.nn.CosineSimilarity(eps=1e-5)
    relu = torch.nn.ReLU()
    norm = grad_zero.norm()
    scores = []
    for i in input_gradients:
        i = i.unsqueeze(0)
        score = cos(i, grad_zero)
        if score < 0:
            score = 0
        scores.append(score)
    scores = torch.tensor([item for item in scores]).to(device)
    # print(scores)
    scores = relu(scores)

    grad = torch.nn.functional.normalize(replica) * norm
    grad = (grad.transpose(0, 1) * scores).transpose(0, 1)
    grad = torch.sum(grad, dim=0) / scores.sum()
    return grad
