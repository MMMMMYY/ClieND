import json
import torch
import os
import pandas as pd
from utils_models import *
from data_utils import *
from attacks import *
from training import *
from utils_file import *
import torch.optim as optim
import sys

# Check if a GPU is available and set the device
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(f"Using device: {device}")

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

num_clients = config['num_clients']
target_class = config.get('target_class', 0)
epochs = config['epochs']
dataset = config['dataset']
batch_size = config['batch_size']
learning_rate = config['learning_rate']
k = config['k']
model_choice = config['model_choice']
attack_config = config.get('attack', {})
attack_enabled = attack_config.get('enabled', False)
attack_type = attack_config.get('type', "min-max")
deviation_type = attack_config.get('deviation_type', 'unit_vec')
num_malicious = attack_config.get('n_attackers', 0)
attack_start_epoch = attack_config.get('attack_start_epoch', 30)
target_flip_labels = attack_config.get('target_flip_labels', 2)
specific_labels_to_flip = attack_config.get('specific_labels_to_flip', None)
severity = attack_config.get('severity', 0.5)
random_seed = attack_config.get('random_seed', None)
separation_factor = attack_config.get('separation_factor', 2.0)
dropout_rate = attack_config.get('dropout_rate', 0.5)
q = config["q"]

file_suffix = f"{model_choice}_{attack_type}_attack-{attack_enabled}_{num_malicious}_attackers_non_iid_{q}"

if not os.path.exists(f'results/{file_suffix}'):
    os.makedirs(f'results/{file_suffix}')
log_file = open(f"results/{file_suffix}/training_log.log", "w")
error_log_file = open(f"results/{file_suffix}/error_log.log", "w")
sys.stdout = log_file
sys.stderr = error_log_file

print(f"Attack configuration: {attack_config}")

# Set random seed for reproducibility
if random_seed is not None:
    torch.manual_seed(random_seed)
print(f"Random seed: {random_seed}")

model_config = config['models'][model_choice]
if model_choice == f"SimpleCNN_{dataset}":
    conv_layers_config = model_config['conv_layers']
    fc_layers_config = model_config['fc_layers']

# Load dataset and create data loaders for clients
train_loaders, val_loader, test_loader = load_dataset(config)
# Determine input dimensions based on dataset type
if dataset in ['mnist', 'cifar10']:
    dataset_type = 'image'
elif dataset == "texas" :
    dataset_type = 'tabular'
elif dataset == "enron":
    dataset_type = 'text'

if dataset_type == 'tabular':
    input_dim = config['models'][f'UnifiedFullyConnectedNN_{dataset}']['input_dim']
    output_dim = config['models'][f'UnifiedFullyConnectedNN_{dataset}']['output_dim']
elif dataset_type == 'text':
    input_dim = config['models'][f'UnifiedFullyConnectedNN_{dataset}']['input_dim']
    output_dim = config['models'][f'UnifiedFullyConnectedNN_{dataset}']['output_dim']
# Set up the unified fully connected model
hidden_dims = [128, 64, 32]  # Example hidden layer sizes
# model = UnifiedFullyConnectedNN(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=1).to(device)
#

# Initialize the global model based on model choice
if model_choice == "SimpleCNN_cifar10":
    global_model = SimpleCNN_cifar10().to(device)
elif model_choice == "SimpleCNN_mnist":
    global_model = SimpleCNN_mnist().to(device)
elif model_choice == "Net":
    global_model = Net(conv_layers_config, fc_layers_config).to(device)
elif model_choice == "UnifiedFullyConnectedNN_texas":
    global_model = TexasFullyConnectedNN().to(device)


criterion = torch.nn.CrossEntropyLoss().to(device)

# Initialize local models and optimizers for each client
local_models = []
client_scores_dict = {}
client_weights_dict = {}
client_accuracies_dict = {}

for i in range(num_clients):
    if model_choice == "SimpleCNN_cifar10":
        base_model = SimpleCNN_cifar10().to(device)
    elif model_choice == "SimpleCNN_mnist":
        base_model = SimpleCNN_mnist().to(device)
    elif model_choice == "Net":
        base_model = Net(conv_layers_config, fc_layers_config).to(device)
    elif model_choice == "UnifiedFullyConnectedNN_texas":
        base_model = TexasFullyConnectedNN().to(device)
    if i < num_malicious and attack_type in ['min_activation', 'sample_dropping', 'neuron_separation']:  # Malicious clients
        model = AttackedModel(base_model, attack_type=attack_type,
                              dropout_rate=dropout_rate,
                              separation_factor=separation_factor).to(device)
    else:
        model = base_model.to(device)
        # print("use base model")

    local_models.append(model)
    client_scores_dict[i] = []
    client_weights_dict[i] = []
    client_accuracies_dict[i] = []

optimizers = [torch.optim.SGD(local_models[i].parameters(), lr=learning_rate) for i in range(num_clients)]

# Initialize lists for storing server scores, weights, and accuracies
server_scores = []
server_weights = []
server_accuracies = []

# Training loop
for epoch in range(epochs):
    trained_local_models = []
    local_grads = []

    # Train local models
    for i in range(num_clients):
        local_models[i].load_state_dict(global_model.state_dict())

        if i < num_malicious:  # Malicious clients
            if attack_enabled and epoch >= attack_start_epoch:
                print(f"Client {i} is malicious!")
                if attack_type in ["min-max", "fang"]:
                    trained_local_model = train(local_models[i], train_loaders[i], criterion, optimizers[i], epochs=1, device=device)

                    # Extract gradients
                    grads = []
                    for param in local_models[i].parameters():
                        if param.grad is not None:
                            grads.append(param.grad.view(-1))

                    if grads:
                        grads = torch.cat(grads).to(device)
                        local_grads.append(grads)

                        if attack_type == "fang":
                            malicious_model = generate_malicious_update_fang(local_models[i], global_model, local_grads,
                                                                            num_malicious, deviation_type)
                        elif attack_type == "min-max":
                            malicious_model = generate_malicious_update(local_models[i], global_model, local_grads,
                                                                        num_malicious, deviation_type)

                        trained_local_models.append(malicious_model)
                    else:
                        trained_local_models.append(trained_local_model)

                elif attack_type == "lie":
                    grads = []
                    for param in local_models[i].parameters():
                        if param.grad is not None:
                            grads.append(param.grad.view(-1))

                    if grads:
                        grads = torch.cat(grads).to(device)
                        local_grads.append(grads)
                        malicious_update = lie_attack(torch.stack(local_grads), z=config['attack'].get('z', 1.0))

                        with torch.no_grad():
                            split_sizes = [param.numel() for param in local_models[i].parameters()]
                            for param, update in zip(local_models[i].parameters(), malicious_update.split(split_sizes)):
                                param.data.copy_(param.data + update.view_as(param))

                    trained_local_models.append(local_models[i])

                elif attack_type in ['min_activation', 'sample_dropping', 'neuron_separation']:
                    trained_local_model = train_dropout(local_models[i], train_loaders[i], criterion, optimizers[i], epochs=1, device=device)
                    trained_local_models.append(trained_local_model)

                elif attack_type == "label_poison":
                    trained_local_models.append(
                        train_malicious(local_models[i], train_loaders[i], criterion, optimizers[i], target_class,
                                        epochs=1, device=device, flip_labels=target_flip_labels,
                                        specific_labels=specific_labels_to_flip))

                elif attack_type == "distributed_attack":
                    trained_local_models.append(train_disributed_attack(local_models[i], train_loaders[i], criterion, optimizers[i], epochs=1, device=device))
                    # poisoned_samples, poisoned_labels = poison_data(train_samples[i], train_labels[i], pdr=0.5)
                    # trained_local_models.train_model(local_models[i], poisoned_samples, poisoned_labels, optimizers[i], criterion)

            else:  # Before attack start or honest clients
                trained_local_models.append(
                    train(local_models[i], train_loaders[i], criterion, optimizers[i], epochs=1, device=device))
        else:  # Honest clients
            trained_local_models.append(
                train(local_models[i], train_loaders[i], criterion, optimizers[i], epochs=1, device=device))
        # Record client scores, weights, and accuracies
        client_scores_dict[i].append(return_score(local_models[i]))
        # client_weights_dict[i].append(return_weight(local_models[i]))
        client_accuracy = test(local_models[i], test_loader, device=device)
        client_accuracies_dict[i].append(client_accuracy)

    # Record server scores, weights, and accuracy
    server_scores.append(return_score(global_model))
    # server_weights.append(return_weight(global_model))
    global_accuracy = test(global_model, test_loader, device=device)
    server_accuracies.append(global_accuracy)

    # Aggregate models
    global_model = average_models(global_model, trained_local_models)

    print(
        f'Epoch {epoch + 1} complete! Client Accuracies: {[f"{acc[-1]:.2f}%" for acc in client_accuracies_dict.values()]} | Global Accuracy: {global_accuracy:.2f}%')

# Save client and server scores, weights, and accuracies
for client_id, scores in client_scores_dict.items():
    df = pd.DataFrame(scores)
    df.to_csv(f'results/{file_suffix}/client_{client_id}_scores.csv', index=False)

# for client_id, weights in client_weights_dict.items():
#     df = pd.DataFrame(weights)
#     df.to_csv(f'results/{file_suffix}/client_{client_id}_weights.csv', index=False)

for client_id, accuracies in client_accuracies_dict.items():
    df = pd.DataFrame(accuracies, columns=['accuracy'])
    df.to_csv(f'results/{file_suffix}/client_{client_id}_accuracies.csv', index=False)

df = pd.DataFrame(server_scores)
df.to_csv(f'results/{file_suffix}/server_scores.csv', index=False)

# df = pd.DataFrame(server_weights)
# df.to_csv(f'results/{file_suffix}/server_weights.csv', index=False)

df = pd.DataFrame(server_accuracies, columns=['accuracy'])
df.to_csv(f'results/{file_suffix}/server_accuracies.csv', index=False)

print("Training complete!")

# Reset stdout to its original state
sys.stdout = sys.__stdout__
log_file.close()
sys.stderr = sys.__stderr__
error_log_file.close()
