import torch
import numpy as np

def average_models(global_model, local_models):
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack([local_model.state_dict()[k].float() for local_model in local_models], 0).mean(0)
    global_model.load_state_dict(global_dict)
    return global_model

def return_score(model):
    """
    Returns the scores for the given model.
    This function assumes that the model has an attribute `popup_scores`
    or similar, which contains the scores for each layer or parameter.

    Args:
        model (torch.nn.Module): The model from which to extract scores.

    Returns:
        list: A list of scores (numpy arrays) for each parameter in the model.
    """
    scores = []
    for name, param in model.named_parameters():
        if 'popup_scores' in name:
            scores.append(param.data.cpu().numpy().flatten())
        elif hasattr(param, 'scores'):
            scores.append(param.scores.data.cpu().numpy().flatten())
    return np.concatenate(scores) if scores else np.array([])


def return_weight(model):
    """
    Returns the weights for the given model.

    Args:
        model (torch.nn.Module): The model from which to extract weights.

    Returns:
        list: A list of weights (numpy arrays) for each parameter in the model.
    """
    weights = []
    for name, param in model.named_parameters():
        weights.append(param.data.cpu().numpy().flatten())
    return np.concatenate(weights)