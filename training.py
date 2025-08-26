import torch
from attacks import *

def train(model, train_loader, criterion, optimizer, epochs=1, device='cpu'):
    model.to(device) 
    model.train()
    for epoch in range(epochs):
        for data, target in train_loader:
            # print(f"Data batch size: {data.size(0)}, Target batch size: {target.size(0)}")
            data, target = data.to(device), target.to(device)  
            optimizer.zero_grad()
            output = model(data)
            # print(output)
            # print(target)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    return model

def train_dropout(model, train_loader, criterion, optimizer, epochs=1, device='cpu'):
    model.to(device)  
    model.train()
    for epoch in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device) 
            optimizer.zero_grad()
            output = model(data, attack_enabled=True)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    return model


def train_malicious(model, train_loader, criterion, optimizer, target_class, epochs=1, device='cpu', flip_labels=2, specific_labels=None):
    model.to(device)  
    model.train()
    for epoch in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device) 
            if specific_labels:
                indices = [i for i, t in enumerate(target) if t.item() in specific_labels]
                flip_indices = indices[:flip_labels]  
            else:
                
                flip_indices = torch.randperm(target.size(0))[:flip_labels]

            target[flip_indices] = target_class  

            output = model(data)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model

def train_disributed_attack(model, train_loader, criterion, optimizer, epochs=1, device='cpu'):
    model.to(device)  # Move the model to the specified device (CPU or GPU)
    model.train()

    for epoch in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)  # Move data and target to the specified device

            # Poison the data based on the poison data rate (pdr)
            poisoned_data, poisoned_target = poison_data(data, target, pdr=0.5)

            output = model(poisoned_data)  # Forward pass
            loss = criterion(output, poisoned_target)  # Calculate loss
            optimizer.zero_grad()  # Zero the gradients
            loss.backward()  # Backpropagation
            optimizer.step()  # Optimize the model
        return model

def test(model, test_loader, device='cpu'):
    model.to(device)  
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)  
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = 100 * correct / total
    return accuracy
