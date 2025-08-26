from cifar10_models import *
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import os
import time
from constant import *

def update_fed_model(model, optimizer, aggregated_grads):
    """
    Update the model with the aggregated parameters
    :param aggregated_parameters: the aggregated parameters
    :return: None
    """
    optimizer.zero_grad()
    model_grads = []
    start_idx = 0
    for i, param in enumerate(model.parameters()):
        param_ = aggregated_grads[start_idx:start_idx + len(param.data.view(-1))].reshape(param.data.shape)
        start_idx = start_idx + len(param.data.view(-1))
        param_ = param_.cuda()
        # param.grad = param_.clone()  # Assign the gradient to the parameter.grad attribute
        model_grads.append(param_)
    optimizer.step(model_grads)

def normal_epoch(model, batch_x, batch_y):
    """
    Train the model for one epoch
    :param batch_x: the input data
    :param batch_y: the target label
    :return: None
    """
    # self.model.train()
    # self.optimizer.zero_grad()
    model.zero_grad()
    output = model(batch_x)
    loss = criterion(output, batch_y)
    loss.backward(retain_graph=True)
    param_grad = []
    for param in model.parameters():
        param_grad = param.grad.data.view(-1) if not len(param_grad) else torch.cat(
            (param_grad, param.grad.view(-1)))


    # self.optimizer.step(param_grad)
    return param_grad

def load_cifar10(data_loc):
    X = []
    Y = []
    train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    cifar10_train = datasets.CIFAR10(root=data_loc, train=True, download=True, transform=train_transform)

    cifar10_test = datasets.CIFAR10(root=data_loc, train=False, download=True, transform=train_transform)

    for i in range(len(cifar10_train)):
        X.append(cifar10_train[i][0].numpy())
        Y.append(cifar10_train[i][1])

    for i in range(len(cifar10_test)):
        X.append(cifar10_test[i][0].numpy())
        Y.append(cifar10_test[i][1])

    X=np.array(X)
    Y=np.array(Y)

    print('total data len: ',len(X))

    if not os.path.isfile('./cifar10_shuffle.pkl'):
        all_indices = np.arange(len(X))
        np.random.shuffle(all_indices)
        pickle.dump(all_indices,open('./cifar10_shuffle.pkl','wb'))
    else:
        all_indices=pickle.load(open('./cifar10_shuffle.pkl','rb'))

    X=X[all_indices]
    Y=Y[all_indices]

    return X,Y




def load_mnist(data_loc):
    X = []
    Y = []
    train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

    mnist_train = datasets.MNIST(root=data_loc, train=True, download=True, transform=train_transform)

    mnist_test = datasets.MNIST(root=data_loc, train=False, download=True, transform=train_transform)

    for i in range(len(mnist_train)):
        X.append(mnist_train[i][0].numpy())
        Y.append(mnist_train[i][1])

    for i in range(len(mnist_test)):
        X.append(mnist_test[i][0].numpy())
        Y.append(mnist_test[i][1])

    X=np.array(X)
    Y=np.array(Y)

    print('total data len: ',len(X))

    if not os.path.isfile('./mnist_shuffle.pkl'):
        all_indices = np.arange(len(X))
        np.random.shuffle(all_indices)
        pickle.dump(all_indices,open('./mnist_shuffle.pkl','wb'))
    else:
        all_indices=pickle.load(open('./mnist_shuffle.pkl','rb'))

    X=X[all_indices]
    Y=Y[all_indices]

    return X,Y

def data_loading(X,Y,nusers,total_tr_len,val_len,te_len,user_tr_len, fltrust_size=fltrust_size, batch_size=batch_size):

    if not os.path.isfile('./cifar10_shuffle.pkl'):
        all_indices = np.arange(len(X))
        np.random.shuffle(all_indices)
        pickle.dump(all_indices, open('./cifar10_shuffle.pkl', 'wb'))
    else:
        all_indices = pickle.load(open('./cifar10_shuffle.pkl', 'rb'))

    total_tr_data = X[:total_tr_len]
    total_tr_label = Y[:total_tr_len]

    val_data = X[total_tr_len:(total_tr_len + val_len)]
    val_label = Y[total_tr_len:(total_tr_len + val_len)]

    fltrust_data = val_data[:fltrust_size]
    fltrust_label = val_label[:fltrust_size]

    val_data = val_data[fltrust_size:]
    val_label = val_label[fltrust_size:]

    te_data = X[(total_tr_len + val_len):(total_tr_len + val_len + te_len)]
    te_label = Y[(total_tr_len + val_len):(total_tr_len + val_len + te_len)]

    total_tr_data_tensor = torch.from_numpy(total_tr_data).type(torch.FloatTensor).to(device)
    total_tr_label_tensor = torch.from_numpy(total_tr_label).type(torch.LongTensor).to(device)

    val_data_tensor = torch.from_numpy(val_data).type(torch.FloatTensor).to(device)
    val_label_tensor = torch.from_numpy(val_label).type(torch.LongTensor).to(device)

    fltrust_data_tensor = torch.from_numpy(fltrust_data).type(torch.FloatTensor).to(device)
    fltrust_label_tensor = torch.from_numpy(fltrust_label).type(torch.LongTensor).to(device)

    te_data_tensor = torch.from_numpy(te_data).type(torch.FloatTensor).to(device)
    te_label_tensor = torch.from_numpy(te_label).type(torch.LongTensor).to(device)

    print('total tr len %d | val len %d | test len %d' % (
    len(total_tr_data_tensor), len(val_data_tensor), len(te_data_tensor)))

    user_tr_data_tensors = []
    user_tr_label_tensors = []

    for i in range(nusers):
        user_tr_data_tensor = torch.from_numpy(total_tr_data[user_tr_len * i:user_tr_len * (i + 1)]).type(
            torch.FloatTensor).to(device)
        user_tr_label_tensor = torch.from_numpy(total_tr_label[user_tr_len * i:user_tr_len * (i + 1)]).type(
            torch.LongTensor).to(device)

        user_tr_data_tensors.append(user_tr_data_tensor)
        user_tr_label_tensors.append(user_tr_label_tensor)
        print('user %d tr len %d' % (i, len(user_tr_data_tensor)))

    #sperate in batch
    # fltrust_data_tensor = fltrust_data_tensor.view(-1, batch_size, 3, 32, 32)
    # fltrust_label_tensor = fltrust_label_tensor.view(-1, batch_size)
    return user_tr_data_tensors, user_tr_label_tensors, val_data_tensor, val_label_tensor, te_data_tensor, te_label_tensor, fltrust_data_tensor, fltrust_label_tensor


def data_loading_class_based(X, Y, nusers, total_tr_len, val_len, te_len, user_tr_len, fltrust_size=fltrust_size, batch_size=batch_size):

    if not os.path.isfile('./cifar10_shuffle.pkl'):
        all_indices = np.arange(len(X))
        np.random.shuffle(all_indices)
        pickle.dump(all_indices, open('./cifar10_shuffle.pkl', 'wb'))
    else:
        all_indices = pickle.load(open('./cifar10_shuffle.pkl', 'rb'))
    Y = torch.tensor(Y).to(device)
    class_count = torch.max(Y).item() + 1
    my_set = [[] for _ in range(nusers)]

    for i in range(class_count):
        user_index = i % nusers
        class_indices = (Y == i).nonzero().squeeze().cpu().numpy()
        class_data = X[class_indices]
        class_labels = Y[class_indices]
        my_set[user_index].append((class_data, class_labels))

    training_set = [[] for _ in range(nusers)]
    val_set = [[] for _ in range(nusers)]
    te_set = [[] for _ in range(nusers)]

    for user_index, user_data in enumerate(my_set):
        for class_data, class_labels in user_data:
            training_index = int(len(class_data) * 0.7)
            test_index = int(len(class_data) * 0.9)

            training_data = class_data[:training_index]
            training_labels = class_labels[:training_index]
            test_data = class_data[training_index:test_index]
            test_labels = class_labels[training_index:test_index]
            val_data = class_data[test_index:]
            val_labels = class_labels[test_index:]

            training_set[user_index].append((training_data, training_labels))
            te_set[user_index].append((test_data, test_labels))
            val_set[user_index].append((val_data, val_labels))

    # print("train", training_set[0][0][0].shape)
    # print(training_set[0][0][1].shape)
    # print(val_set[0][0][0].shape)

    tr_data_tensors = []
    tr_label_tensors = []

    for train_data in training_set:
        if train_data:
            tr_data_tensor = torch.cat([torch.tensor(data[0]).clone().detach() for data in train_data]).to(device)
            tr_label_tensor = torch.cat([torch.tensor(data[1]).clone().detach() for data in train_data]).to(device)
            tr_data_tensors.append(tr_data_tensor)
            tr_label_tensors.append(tr_label_tensor)

    # convert to array
    # tr_data_tensors = np.array(tr_data_tensors.)
    # tr_label_tensors = np.array(tr_label_tensors)
    val_data_tensors = []
    val_label_tensors = []

    for val_data in val_set:
        if val_data:
            val_data_tensor = torch.cat([torch.tensor(data[0]).clone().detach() for data in val_data]).to(device)
            val_label_tensor = torch.cat([torch.tensor(data[1]).clone().detach() for data in val_data]).to(device)
            val_data_tensors.append(val_data_tensor)
            val_label_tensors.append(val_label_tensor)

    te_data_tensors = []
    te_label_tensors = []

    for te_data in te_set:
        if te_data:
            te_data_tensor = torch.cat([torch.tensor(data[0]).clone().detach() for data in te_data]).to(device)
            te_label_tensor = torch.cat([torch.tensor(data[1]).clone().detach() for data in te_data]).to(device)
            te_data_tensors.append(te_data_tensor)
            te_label_tensors.append(te_label_tensor)

    if val_data_tensors:
        fltrust_data_tensor = val_data_tensors[0][:fltrust_size].to(device)
        fltrust_label_tensor = val_label_tensors[0][:fltrust_size].to(device)
    else:
        fltrust_data_tensor = torch.empty(0).to(device)
        fltrust_label_tensor = torch.empty(0).to(device)

    user_tr_data_tensors = []
    user_tr_label_tensors = []

    for i in range(nusers):
        start_index = user_tr_len * i
        end_index = user_tr_len * (i + 1)
        # Concatenate tensors in the list into single tensors
        user_tr_data_tensor = torch.cat([data_tensor[start_index:end_index] for data_tensor in tr_data_tensors],
                                        dim=0).to(device)
        user_tr_label_tensor = torch.cat([label_tensor[start_index:end_index] for label_tensor in tr_label_tensors],
                                         dim=0).to(device)

        user_tr_data_tensors.append(user_tr_data_tensor)
        user_tr_label_tensors.append(user_tr_label_tensor)



    print(len(user_tr_data_tensors[0]), len(val_data_tensors[0]), len(te_data_tensors[0]))

    return user_tr_data_tensors, user_tr_label_tensors, val_data_tensors, val_label_tensors, te_data_tensors, te_label_tensors, fltrust_data_tensor, fltrust_label_tensor


def cifar10_noniid(data_loc, num_users, val_len, te_len, batch_size = batch_size):
    """
    Sample non-I.I.D client data from CIFAR-10 dataset
    :param dataset: CIFAR-10 dataset
    :param num_users: Number of users
    :param val_len: Length of the validation set
    :param te_len: Length of the test set
    :param batch_size: Batch size
    :param device: Device to store the tensors
    :return: Tensors for training, validation, and test sets
    
    
    """

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    dataset = datasets.CIFAR10(root=data_loc, train=True, download=True, transform=train_transform)

    num_shards, num_imgs = 200, 250  # Each random user receives (2*250) samples
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = np.array(dataset.targets)

    # Sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)

    user_tr_data_tensors = []
    user_tr_label_tensors = []
    val_data_tensors = []
    val_label_tensors = []
    te_data_tensors = []
    te_label_tensors = []
    tr_data_tensors = []
    tr_label_tensors = []

    for user, indices in dict_users.items():
        # Shuffle indices
        np.random.shuffle(indices)

        # Split indices into training, validation, and test sets
        train_indices = indices[:-val_len - te_len]
        val_indices = indices[-val_len - te_len:-te_len]
        test_indices = indices[-te_len:]

        # Extract data and labels
        train_data = torch.tensor(dataset.data[train_indices]).permute(0, 3, 1, 2).float().to(device)
        train_labels = torch.tensor(np.array(dataset.targets)[train_indices]).long().to(device)
        val_data = torch.tensor(dataset.data[val_indices]).permute(0, 3, 1, 2).float().to(device)
        val_labels = torch.tensor(np.array(dataset.targets)[val_indices]).long().to(device)
        test_data = torch.tensor(dataset.data[test_indices]).permute(0, 3, 1, 2).float().to(device)
        test_labels = torch.tensor(np.array(dataset.targets)[test_indices]).long().to(device)

        tr_data_tensors.append(train_data)
        tr_label_tensors.append(train_labels)
        val_data_tensors.append(val_data)
        val_label_tensors.append(val_labels)
        te_data_tensors.append(test_data)
        te_label_tensors.append(test_labels)

        for i in range(nusers):
            start_index = user_tr_len * i
            end_index = user_tr_len * (i + 1)
            # Concatenate tensors in the list into single tensors
            user_tr_data_tensor = torch.cat([data_tensor[start_index:end_index] for data_tensor in tr_data_tensors],
                                            dim=0).to(device)
            user_tr_label_tensor = torch.cat([label_tensor[start_index:end_index] for label_tensor in tr_label_tensors],
                                             dim=0).to(device)

            user_tr_data_tensors.append(user_tr_data_tensor)
            user_tr_label_tensors.append(user_tr_label_tensor)

    # Use the validation set as the FLTrust data
    fltrust_data_tensor = torch.cat([batch for user_batches in val_data_tensors for batch in user_batches]).to(device)
    fltrust_label_tensor = torch.cat([batch for user_batches in val_label_tensors for batch in user_batches]).to(device)


    return user_tr_data_tensors, user_tr_label_tensors, val_data_tensors, val_label_tensors, te_data_tensors, te_label_tensors, fltrust_data_tensor, fltrust_label_tensor


def train(train_data, labels, model, criterion, optimizer, use_cuda, num_batchs=999999, debug_='MEDIUM', batch_size=16):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    len_t = (len(train_data) // batch_size) - 1

    for ind in range(len_t):
        if ind > num_batchs:
            break
        # measure data loading time
        inputs = train_data[ind * batch_size:(ind + 1) * batch_size]
        targets = labels[ind * batch_size:(ind + 1) * batch_size]

        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)

        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        if debug_ == 'HIGH' and ind % 100 == 0:
            print('Classifier: ({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                batch=ind + 1,
                size=len_t,
                data=data_time.avg,
                bt=batch_time.avg,
                loss=losses.avg,
                top1=top1.avg,
                top5=top5.avg,
            ))

    return (losses.avg, top1.avg)


# In[9]:


def test(test_data, labels, model, criterion, use_cuda, debug_='MEDIUM', batch_size=64):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    len_t = (len(test_data) // batch_size) - 1

    with torch.no_grad():
        for ind in range(len_t):
            # measure data loading time
            inputs = test_data[ind * batch_size:(ind + 1) * batch_size]
            targets = labels[ind * batch_size:(ind + 1) * batch_size]

            data_time.update(time.time() - end)

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

            # compute output
            outputs = model(inputs)

            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            if debug_ == 'HIGH' and ind % 100 == 0:
                print('Test classifier: ({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=ind + 1,
                    size=len(test_data),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                ))

    return (losses.avg, top1.avg)

def update_opt(model,agg_grads):
    start_idx = 0
    model_grads = []
    for i, param in enumerate(model.parameters()):
        param_ = agg_grads[start_idx:start_idx + len(param.data.view(-1))].reshape(param.data.shape)
        start_idx = start_idx + len(param.data.view(-1))
        param_ = param_.cuda()
        model_grads.append(param_)
    return model_grads
