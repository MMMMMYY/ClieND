import torch
import torch.nn.functional as F
from torch import autograd
import numpy as np

class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int((1 - k) * scores.numel())
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1
        return out

    @staticmethod
    def backward(ctx, g):
        return g, None

def our_attack_dist(all_updates, model_re, n_attackers, dev_type='unit_vec'):
    device = model_re.device  # Ensure all tensors are on the same device as model_re

    if dev_type == 'unit_vec':
        deviation = model_re / torch.norm(model_re).to(device)  # Ensure norm is on the same device
    elif dev_type == 'sign':
        deviation = torch.sign(model_re).to(device)
    elif dev_type == 'std':
        deviation = torch.std(all_updates, 0).to(device)

    lamda = torch.Tensor([10.0]).float().to(device)  # Move lambda to the correct device
    threshold_diff = 1e-5
    lamda_fail = lamda.clone()
    lamda_succ = torch.tensor(0.0, device=device)

    distances = []
    for update in all_updates:
        distance = torch.norm((all_updates - update).to(device), dim=1) ** 2
        distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)

    max_distance = torch.max(distances).to(device)
    del distances

    while torch.abs(lamda_succ - lamda) > threshold_diff:
        mal_update = (model_re - lamda * deviation).to(device)
        distance = torch.norm((all_updates - mal_update).to(device), dim=1) ** 2
        max_d = torch.max(distance).to(device)

        if max_d <= max_distance:
            lamda_succ = lamda.clone()
            lamda = lamda + lamda_fail / 2
        else:
            lamda = lamda - lamda_fail / 2

        lamda_fail = lamda_fail / 2

    mal_update = (model_re - lamda_succ * deviation).to(device)

    return mal_update


def generate_malicious_update(local_model, global_model, local_grads, n_attackers, dev_type='unit_vec'):
    device = next(local_model.parameters()).device  

    local_grads_tensor = torch.stack([grad.to(device) for grad in local_grads])


    agg_grads = torch.mean(local_grads_tensor, 0)


    malicious_update = our_attack_dist(local_grads_tensor, agg_grads, n_attackers, dev_type)


    with torch.no_grad():
        for param, mal_update in zip(local_model.parameters(), malicious_update):
            mal_update = mal_update.to(device) 
            param.copy_(param - mal_update)

    return local_model




def generate_malicious_update_fang(local_model, global_model, local_grads, n_attackers, dev_type='unit_vec'):
    """
    Generates a malicious update using the Fang attack method.

    Args:
        local_model (torch.nn.Module): The local model of the malicious client.
        global_model (torch.nn.Module): The global model to be attacked.
        local_grads (list of torch.Tensor): List of local gradients from malicious clients.
        n_attackers (int): Number of attackers.
        dev_type (str): Type of deviation, e.g., 'unit_vec'.

    Returns:
        torch.nn.Module: The local model updated with the malicious gradient.
    """

    # Step 1: Compute the average of the local gradients (benign gradients)
    benign_grads = torch.mean(torch.stack(local_grads), dim=0)

    # Step 2: Generate deviation vector
    if dev_type == 'unit_vec':
        # Generate a unit vector with the same direction as the benign gradient
        deviation = torch.sign(benign_grads)
    else:
        # Custom deviation type can be handled here
        deviation = torch.sign(benign_grads)  # Default to unit vector

    # Step 3: Scale the deviation to create a malicious gradient
    mal_update = -deviation * torch.max(torch.abs(benign_grads))

    # Step 4: Apply the malicious update to the model parameters
    with torch.no_grad():
        for param, mal_grad in zip(local_model.parameters(), mal_update):
            param.copy_(param + mal_grad)

    return local_model


def lie_attack(all_updates, z):
    """
    Performs the LIE (Little Is Enough) attack.

    Args:
        all_updates (torch.Tensor): Tensor containing all the updates from clients.
        z (float): The scaling factor for the standard deviation.

    Returns:
        torch.Tensor: The malicious update.
    """
    avg = torch.mean(all_updates, dim=0)
    std = torch.std(all_updates, dim=0)

    # Ensure std has no zeros to prevent invalid values
    std[std == 0] = 1e-8

    # Compute the malicious update
    malicious_update = avg + z * std

    # Ensure the malicious update matches the shape of the parameters
    if malicious_update.numel() != all_updates.shape[1]:
        raise RuntimeError(
            f"Malicious update shape {malicious_update.shape} does not match expected shape {all_updates.shape[1:]}")

    return malicious_update

# def trigger_single_image(image):
#     """
#     Adds a red square with a height/width of 6 pixels into
#     the upper left corner of the given image.
#     :param image tensor, containing the normalized pixel values of the image.
#     The image will be modified in-place.
#     :return given image
#     """
#     mean = torch.Tensor([0.485, 0.456, 0.406])
#     std_dev = torch.Tensor([0.229, 0.224, 0.225])
#     color = (torch.Tensor((1, 0, 0)) - mean) / std_dev
#     image[:, 0:6, 0:6] = color.repeat((6, 6, 1)).permute(2, 1, 0)
#     return image
def trigger_single_image(image, trigger_size=6, color=None):
    """
    Add a trigger pattern to the image. The trigger is a small square in the top-left corner of the image.
    The color of the trigger is determined by the 'color' parameter.
    If the image has 3 channels (e.g., RGB), the color will be repeated across all channels.
    """
    # If no color is provided, default to a specific color pattern
    if color is None:
        color = torch.tensor([1.0])  # Default color (you can change this)

    # Determine the number of channels in the image
    num_channels = image.shape[0]

    # If the image has 3 channels (e.g., RGB), repeat the color across all channels
    if num_channels == 3 and color.shape[0] == 1:
        color = color.repeat(num_channels)

    # Ensure the color tensor matches the number of channels in the image
    color = color.view(num_channels, 1, 1)  # Shape it for broadcasting

    # Apply the trigger to the top-left corner of the image
    image[:, 0:trigger_size, 0:trigger_size] = color.repeat(1, trigger_size, trigger_size)

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

