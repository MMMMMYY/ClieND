import json
import torch
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import numpy as np


# def create_dirichlet_distribution(dataset, num_clients, q):
#     labels = np.array([dataset.labels[idx] for idx in range(len(dataset))])
#     label_distribution = np.bincount(labels)
#
#     print(f"Label distribution: {label_distribution}")
#     print(f"Total samples in dataset: {len(dataset)}")
#
#     client_datasets = []
#     for i in range(num_clients):
#         alpha = q * label_distribution / sum(label_distribution)
#         alpha[alpha <= 0] = 1e-3  # 平滑因子，确保 alpha 没有小于或等于0的值
#         proportions = np.random.dirichlet(alpha + 1e-3)
#
#         print(f"Proportions for client {i}: {proportions}")
#
#         num_samples = int(proportions[i] * len(dataset))
#         print(f"Number of samples for client {i}: {num_samples}")
#
#         indices = np.random.choice(len(dataset), num_samples, replace=False)
#         print(f"Indices for client {i}: {indices}")
#
#         if len(indices) > 0:
#             client_datasets.append(Subset(dataset, indices))
#         else:
#             print(f"Warning: Client {i} has 0 samples.")
#
#     return list(client_datasets)


# def load_dataset(config):
#     # print(config['datasets']['enron']['data_dir'])
#     dataset_choice = config['dataset']
#     num_clients = config['num_clients']
#     batch_size = config['batch_size']
#     q = config.get('q', 1.0)  # Get the Dirichlet parameter from config or set a default value
#
#     if dataset_choice == 'cifar100':
#         # Load the CIFAR-100 dataset and apply Dirichlet distribution
#         train_loader, val_loader, test_loader = load_cifar100_data(config['datasets']['cifar100'], batch_size)
#         client_datasets = create_dirichlet_distribution(train_loader.dataset, num_clients, q)
#         client_train_loaders = [DataLoader(client_dataset, batch_size=batch_size, shuffle=True) for client_dataset in client_datasets]
#         return client_train_loaders, val_loader, test_loader
#
#     elif dataset_choice == 'mnist':
#         # Load the MNIST dataset and apply Dirichlet distribution
#         train_loader, val_loader, test_loader = load_mnist_data(config['datasets']['mnist'], batch_size)
#         client_datasets = create_dirichlet_distribution(train_loader.dataset, num_clients, q)
#         client_train_loaders = [DataLoader(client_dataset, batch_size=batch_size, shuffle=True) for client_dataset in client_datasets]
#         return client_train_loaders, val_loader, test_loader
#
#     elif dataset_choice == 'texas':
#         # Load the Texas dataset and apply Dirichlet distribution
#         train_loader, val_loader, test_loader = load_texas_hospital_data(config['datasets']['texas'], batch_size)
#         client_datasets = create_dirichlet_distribution(train_loader.dataset, num_clients, q)
#         client_train_loaders = [DataLoader(client_dataset, batch_size=batch_size, shuffle=True) for client_dataset in
#                                 client_datasets]
#         return client_train_loaders, val_loader, test_loader
#
#     elif dataset_choice == 'enron':
#         # Load the Enron dataset and apply Dirichlet distribution
#         train_loader, val_loader, test_loader = load_enron_email_data(config['datasets']['enron'], batch_size)
#         client_datasets = create_dirichlet_distribution(train_loader.dataset, num_clients, q)
#         print(f"Client {i} dataset size: {len(client_dataset)}" for i, client_dataset in enumerate(client_datasets))
#         client_train_loaders = [DataLoader(client_dataset, batch_size=batch_size, shuffle=True) for client_dataset in
#                                 client_datasets]
#         return client_train_loaders, val_loader, test_loader

def load_dataset(config):
    dataset_choice = config['dataset']
    num_clients = config['num_clients']
    batch_size = config['batch_size']

    if dataset_choice == 'cifar100':
        return split_dataset_among_clients(load_cifar100_data, config['datasets']['cifar100'], num_clients, batch_size)

    elif dataset_choice == 'mnist':
        return split_dataset_among_clients(load_mnist_data, config['datasets']['mnist'], num_clients, batch_size)

    elif dataset_choice == 'texas':
        return split_dataset_among_clients(load_texas_hospital_data, config['datasets']['texas'], num_clients, batch_size)

    elif dataset_choice == 'enron':
        return split_dataset_among_clients(load_enron_email_data, config['datasets']['enron'], num_clients, batch_size)

def create_iid_distribution(dataset, num_clients):
    num_samples_per_client = len(dataset) // num_clients
    indices = np.random.permutation(len(dataset))

    client_datasets = []
    for i in range(num_clients):
        client_indices = indices[i * num_samples_per_client: (i + 1) * num_samples_per_client]
        client_datasets.append(Subset(dataset, client_indices))

    # If there are leftover samples, add them to the last client
    if len(dataset) % num_clients != 0:
        client_datasets[-1] = Subset(dataset, indices[(num_clients - 1) * num_samples_per_client:])

    return client_datasets


def create_dirichlet_distribution(dataset, num_clients, q):
    label_distribution = np.bincount(dataset.targets)  # Assumes targets is a numpy array of labels
    client_datasets = []

    for _ in range(num_clients):
        proportions = np.random.dirichlet(q * label_distribution / sum(label_distribution))
        client_indices = np.hstack(
            [np.random.choice(np.where(dataset.targets == label)[0], int(proportion * len(dataset)), replace=False) for
             label, proportion in enumerate(proportions)])
        client_datasets.append(Subset(dataset, client_indices))

    return client_datasets


def split_dataset_among_clients(dataset_loader_func, dataset_config, num_clients, batch_size):
    # Load the entire dataset
    train_loader, val_loader, test_loader = dataset_loader_func(**dataset_config, batch_size=batch_size)

    # Split the training dataset among clients
    train_dataset = train_loader.dataset
    dataset_len = len(train_dataset)
    indices = list(range(dataset_len))
    split_size = dataset_len // num_clients
    client_datasets = [Subset(train_dataset, indices[i * split_size:(i + 1) * split_size]) for i in range(num_clients)]

    # Create DataLoaders for each client
    client_train_loaders = [DataLoader(client_dataset, batch_size=batch_size, shuffle=True) for client_dataset in client_datasets]

    return client_train_loaders, val_loader, test_loader



def load_cifar100_data(train_data_path, test_data_path, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = datasets.CIFAR100(root=train_data_path, train=True, download=True, transform=transform)
    testset = datasets.CIFAR100(root=test_data_path, train=False, download=True, transform=transform)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return train_loader, None, test_loader


def load_mnist_data(train_data_path, test_data_path, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    trainset = datasets.MNIST(root=train_data_path, train=True, download=True, transform=transform)
    testset = datasets.MNIST(root=test_data_path, train=False, download=True, transform=transform)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return train_loader, None, test_loader


def load_texas_hospital_data(npz_file, batch_size):
    dataset = TexasHospitalDataset(npz_file=npz_file)

    train_indices, test_indices = train_test_split(range(len(dataset)), test_size=0.2)
    train_indices, val_indices = train_test_split(train_indices, test_size=0.1)

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader



def load_enron_email_data(data_dir, batch_size):
    df = load_enron_dataset(data_dir)
    X, y, vectorizer, label_encoder = preprocess_enron_dataset(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_dataset = EnronEmailDataset(X_train, y_train)
    # print(f"Total samples in dataset: {len(train_dataset)}")
    test_dataset = EnronEmailDataset(X_test, y_test)
    # print(f"Total samples in test dataset: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, None, test_loader


def load_enron_dataset(data_dir):
    # 提取路径字符串
    data_dir = data_dir['data_dir']

    emails = []
    labels = []

    # print(f"Starting traversal in: {data_dir}")  # 打印出路径以确保它指向正确的目录

    for root, dirs, files in os.walk(data_dir):
        # print(f"Current directory: {root}")  # 打印当前遍历的目录
        for file in files:
            # print(f"Found file: {file}")  # 打印找到的文件名
            if file.endswith("."):  # 根据实际文件类型修改条件
                label = os.path.basename(root)
                file_path = os.path.join(root, file)

                with open(file_path, 'r', encoding='latin1') as f:
                    emails.append(f.read())
                    labels.append(label)

    df = pd.DataFrame({'email': emails, 'label': labels})
    return df




def preprocess_enron_dataset(df):
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['label'])

    vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
    X = vectorizer.fit_transform(df['email']).toarray()
    y = df['label'].values

    return X, y, vectorizer, label_encoder


class TexasHospitalDataset(Dataset):
    def __init__(self, npz_file, device='cpu'):
        self.device = device

        # Load the dataset
        data = np.load(npz_file)
        self.features = data['features']  # Assuming 'features' key contains features
        self.labels = data['labels']      # Assuming 'labels' key contains labels

        # Convert labels to the appropriate format
        self.labels = np.argmax(self.labels, axis=1)
        self.labels = torch.tensor(self.labels, dtype=torch.int64).to(self.device)
        self.features = torch.tensor(self.features, dtype=torch.float).to(self.device)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]



class EnronEmailDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        # self.targets = self.labels  # 将 targets 设置为类的属性

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

