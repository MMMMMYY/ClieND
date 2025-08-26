import torchvision.transforms as transforms
import torchvision.datasets as datasets
import collections
import torch
from torch.utils.data import DataLoader, TensorDataset
from cifar10_models import *
import numpy as np
import os
import pickle
from utils.logger import *
from utils.eval import *
from utils.misc import *
from utils.adv_train import *
from utils.semisup import *
from cifar10_normal_train import *
from cifar10_util import *
from adam import Adam
from sgd import SGD
from attacks import *
import copy
from score_monitor import *
from models import *
from model import TargetModel
from constant import *
import sys
import pandas as pd
from dct import *
from utils.plot import *
from aggregator import *
from utils.server_defense import *
from utils.existing_attack import *
import cProfile





def main():
    # Redirect stderr and stdout
    sys.stderr = open("err.txt", 'w')
    sys.stdout = open("result/{}_{}_{}_{}_{}_{}_{}_out.txt".format(dataset, aggregation, at_type, detection, k, interval, n_attackers[0]), 'w')

    # Load CIFAR-10 data
    data, labels = load_cifar10(cifar_loc)
    if distribution == 'iid':
        user_tr_data_tensors, user_tr_label_tensors, val_data_tensor, val_label_tensor, te_data_tensor, te_label_tensor, fltrust_data, fltrust_label = data_loading(
            data, labels, nusers, total_tr_len, val_len, te_len, user_tr_len)
    elif distribution == 'non-iid':
        user_tr_data_tensors, user_tr_label_tensors, val_data_tensor, val_label_tensor, te_data_tensor, te_label_tensor, fltrust_data, fltrust_label = cifar10_noniid(cifar_loc, nusers, val_len, te_len)

    # Set device to GPU if available
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_data_tensor, val_label_tensor, te_data_tensor, te_label_tensor, fltrust_data, fltrust_label = val_data_tensor.to(device), val_label_tensor.to(device), te_data_tensor.to(device), te_label_tensor.to(device), fltrust_data.to(device), fltrust_label.to(device)

    # Initialize variables
    rank_change_history = []
    acc_history = []

    score_hist = {}
    scores = []
    scores_sum = []
    scores_sum_topk = []
    score_diff = []
    score_diff_topk = []
    horizontals = []
    verticals = []
    last_score_sum = 0
    last_score_sum_topk = 0
    epoch_num = 0
    best_global_acc = 0
    best_global_te_acc = 0
    attacker_join = False
    n_attacker = 0
    lamda_sum = 0

    horizontal_results_list = []
    vertical_results_list = []
    weights_results_list = []
    scores_results_list = []

    # Initialize dictionaries with None
    client_scores_dict = {client_id: None for client_id in range(nusers)}
    client_weights_dict = {client_id: None for client_id in range(nusers)}
    client_models = {client_id: return_model(arch, 0.1, 0.9, parallel=False)[0].to(device) for client_id in
                     range(nusers)}
    client_optimizers = {client_id: SGD(client_models[client_id].parameters(), lr=0.1) for client_id in range(nusers)}

    # Training loop
    for z in z_values:
        fed_file = 'alexnet_checkpoint_%s_%s_%d_%.2f.pth.tar' % (aggregation, at_type, n_attacker, z)
        fed_best_file = 'alexnet_best_%s_%s_%d_%.2f.pth.tar' % (aggregation, at_type, n_attacker, z)

        # if resume:
        #     fed_checkpoint = chkpt + '/' + fed_file
        #     assert os.path.isfile(fed_checkpoint), 'Error: no user checkpoint at %s' % (fed_checkpoint)
        #     checkpoint = torch.load(fed_checkpoint, map_location=device)
        #     fed_model.load_state_dict(checkpoint['state_dict'])
        #     optimizer_fed.load_state_dict(checkpoint['optimizer'])
        #     resume = 0
        #     best_global_acc = checkpoint['best_acc']
        #     best_global_te_acc = checkpoint['best_te_acc']
        #     val_loss, val_acc = test(val_data_tensor, val_label_tensor, fed_model, criterion, use_cuda)
        #     epoch_num += checkpoint['epoch']
        #     print('resuming from epoch %d | val acc %.4f | best acc %.3f | best te acc %.3f' % (
        #         epoch_num, val_acc, best_global_acc, best_global_te_acc))

        torch.cuda.empty_cache()
        fed_model, _ = return_model(arch, 0.1, 0.9, parallel=False)
        fed_model.to(device)
        optimizer_fed = SGD(fed_model.parameters(), lr=0.1)
        # last_ckpt = copy.deepcopy(fed_model.state_dict())
        server_scores = return_score(fed_model)
        server_weights = return_weight(fed_model)
        param_grad = []

        for param in fed_model.parameters():
            if param.grad is not None:
                param_grad.append(param.grad.data.view(-1).clone().detach().cpu())
            else:
                param_grad.append(torch.zeros_like(param.data.view(-1)).cpu())

        param_grad = torch.cat(param_grad).to(device)
        update_fed_model(fed_model, optimizer_fed, param_grad)
        global_score = return_score(fed_model)

        while epoch_num <= nepochs:
            if aggregation == 'FRL' :
                initial_scores = {}
                for n, m in fed_model.named_modules():
                    if hasattr(m, "popup_scores"):
                        initial_scores[str(n)] = m.popup_scores.detach().clone().flatten().sort()[0]
            user_grads = []
            cur_user_updates = collections.defaultdict(list)

            # participant = []
            client_weight = []
            # client_data = []
            # client_label = []
            server_score = return_score(fed_model)
            server_weight = return_weight(fed_model)
            server_scores = np.vstack((server_scores, server_score))
            server_weights = np.vstack((server_weights, server_weight))

            param_grad = []

            for param in fed_model.parameters():
                if param.grad is not None:
                    param_grad.append(param.grad.data.view(-1).clone().detach().cpu())
                else:
                    param_grad.append(torch.zeros_like(param.data.view(-1)).cpu())

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
            # for i in range(nusers):
                # client_models[i] = apply_gradients(client_models[i], param_grad)

            # print(client_scores_dict[0])
            for i in range(n_attacker, nusers):
                inputs = user_tr_data_tensors[i][(epoch_num % nbatches) * batch_size:((epoch_num % nbatches) + 1) * batch_size]
                targets = user_tr_label_tensors[i][(epoch_num % nbatches) * batch_size:((epoch_num % nbatches) + 1) * batch_size]

                if at_type == 'poisoning' and i < n_attackers[0]:
                    if attacker_join:
                        inputs, targets = get_label_flipping(inputs, targets)
                if at_type == 'pixel_attack' and i < n_attackers[0]:
                    if attacker_join:
                        inputs, targets = poison_data(inputs, targets)
                inputs, targets = inputs.to(device), targets.to(device)
                inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
                outputs = client_models[i](inputs)
                loss = criterion(outputs, targets)
                client_models[i].zero_grad()
                loss.backward(retain_graph=True)
                
                
                if aggregation == 'flame':
                    index_participant = i - n_attacker
                    pre_model = {}
                    for name, param in fed_model.state_dict().items():
                        pre_model[name] = param.clone()

                param_grad = []
                for param in client_models[i].parameters():
                    if param.grad is not None:
                        param_grad.append(param.grad.data.view(-1).clone().detach().cpu())
                    else:
                        param_grad.append(torch.zeros_like(param.data.view(-1)).cpu())

                param_grad = torch.cat(param_grad).to(device)
                # val_loss, val_acc = test(val_data_tensor, val_label_tensor, client_models[i], criterion, use_cuda)
                te_loss, te_acc_client_b = test(te_data_tensor, te_label_tensor, client_models[i], criterion, use_cuda)
                update_fed_model(client_models[i], client_optimizers[i], param_grad)
                te_loss, te_acc_client_a = test(te_data_tensor, te_label_tensor, client_models[i], criterion, use_cuda)
                print('client %d te acc before %.4f after %.4f' % (i, te_acc_client_b, te_acc_client_a))

                user_grads = param_grad[None, :] if len(user_grads) == 0 else torch.cat((user_grads, param_grad[None, :]), 0)

                torch.cuda.empty_cache()
                if aggregation == 'flame':
                    update_fed_model(fed_model, optimizer_fed, param_grad)

                    diff = dict()
                    for name, data in fed_model.state_dict().items():
                        diff[name] = (data - pre_model[name])
                    client_weight.append(diff)

                # client_model, _ = return_model(arch, 0.1, 0.9, parallel=False)

                # def apply_gradients(model, grads):
                #     model.zero_grad()
                #     grad_tensors = torch.cat([grad.view(-1) for grad in grads])
                #     start_idx = 0
                #     for param in model.parameters():
                #         end_idx = start_idx + param.numel()
                #         if param.grad is None:
                #             param.grad = torch.zeros_like(param.data)
                #         param.grad.data.copy_(grad_tensors[start_idx:end_idx].view_as(param))
                #         start_idx = end_idx
                #     return model

                # client_models[i] = apply_gradients(client_models[i], param_grad)

                # if epoch_num <= 10:
                #     min_score, max_score = get_the_threshold(client_models[i])
                #     topk_neurons = get_topk_neurons(client_models[i], k)
                #     min_score_topk, max_score_topk = get_the_threshold_topk(client_models[i], topk_neurons)

                # if epoch_num >= 30:
                #     sum_of_detected_neurons = check_neuron_score(client_models[i], min_score, max_score)
                #     sum_of_detected_neurons_topk = check_neuron_score_topk(client_models[i], topk_neurons, min_score_topk, max_score_topk)

                weight = return_weight(client_models[i])
                score = return_score(client_models[i])


                if epoch_num % interval == 0:
                    # if epoch_num >= 30:
                    #     horizontal_results_list.append({"epoch": epoch_num, "client": i, "detected_neurons": sum_of_detected_neurons})
                    #     vertical_results_list.append({"epoch": epoch_num, "client": i, "detected_neurons_topk": sum_of_detected_neurons_topk})
                    # # Save each neuron's score/weight in a separate column
                    weights_results_list.append({"epoch": epoch_num, "client": i, **{"neuron_" + str(j): weight[j] for j in range(len(weight))}})
                    scores_results_list.append({"epoch": epoch_num, "client": i, **{"neuron_" + str(j): score[j] for j in range(len(score))}})

                # Stack scores and weights
                if client_scores_dict[i] is None:
                    client_scores_dict[i] = score
                else:
                    client_scores_dict[i] = np.vstack((client_scores_dict[i], score))

                if client_weights_dict[i] is None:
                    client_weights_dict[i] = weight
                else:
                    client_weights_dict[i] = np.vstack((client_weights_dict[i], weight))
            malicious_grads = user_grads

            if epoch_num == attack_start_epoch - 1:
                attacker_join = True
                if at_type != 'poisoning' and at_type != 'pixel_attack':
                    n_attacker = n_attackers[0]


            if n_attacker > 0 and attacker_join:
                if at_type == 'lie':
                    mal_update = lie_attack(malicious_grads, z_values[n_attacker])
                    malicious_grads = torch.cat((torch.stack([mal_update] * n_attacker), malicious_grads)).to(device)
                elif at_type == 'fang':
                    agg_grads = torch.mean(malicious_grads, 0)
                    deviation = torch.sign(agg_grads)
                    malicious_grads = get_malicious_updates_fang_trmean(malicious_grads, deviation, n_attacker, epoch_num)
                elif at_type == 'our-agr':
                    agg_grads = torch.mean(malicious_grads, 0)
                    mal_update = our_attack_median(malicious_grads, agg_grads, n_attacker, dev_type)
                elif at_type == 'min-max':
                    dev_type = 'unit_vec'
                    agg_grads = torch.mean(malicious_grads, 0)
                    mal_update = our_attack_dist(malicious_grads, agg_grads, n_attacker, dev_type)
                elif at_type == 'min-sum':
                    agg_grads = torch.mean(malicious_grads, 0)
                    mal_update = our_attack_score(malicious_grads, agg_grads, n_attacker, dev_type)
                elif at_type == 'poisoning' or at_type == 'none':
                    mal_update = torch.mean(malicious_grads, 0)
                elif at_type == 'pixel_attack':
                    mal_update = torch.mean(malicious_grads, 0)
                elif at_type == 'rank_reverse':
                    cur_user_updates = rank_reverse(fed_model, cur_user_updates, n_attacker)
                elif at_type == 'dropout_attack':
                    for i in range(n_attacker):
                        inputs = user_tr_data_tensors[i][(epoch_num % nbatches) * batch_size:((epoch_num % nbatches) + 1) * batch_size]
                        targets = user_tr_label_tensors[i][(epoch_num % nbatches) * batch_size:((epoch_num % nbatches) + 1) * batch_size]

                        inputs, targets = inputs.to(device), targets.to(device)
                        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
                        fed_model.mask_top_k_weights(at_k)
                        outputs = fed_model(inputs)
                        loss = criterion(outputs, targets)
                        fed_model.zero_grad()
                        loss.backward(retain_graph=True)
                        torch.cuda.empty_cache()
                        param_grad = []
                        for param in fed_model.parameters():
                            if param.grad is not None:
                                param_grad.append(param.grad.data.view(-1).clone().detach().cpu())
                            else:
                                param_grad.append(torch.zeros_like(param.data.view(-1)).cpu())

                        param_grad = torch.cat(param_grad).to(device)

                        malicious_grads = param_grad[None, :] if len(malicious_grads) == 0 else torch.cat((malicious_grads, param_grad[None, :]), 0)
                    mal_update = malicious_grads
                    attacker_join = False

            if not epoch_num:
                print(malicious_grads.shape)

            if aggregation == 'median':
                agg_grads = torch.median(malicious_grads, dim=0).to(device)[0]
                update_fed_model(fed_model, optimizer_fed, agg_grads)
            elif aggregation == 'trmean':
                agg_grads = tr_mean(malicious_grads, n_attacker).to(device)
                update_fed_model(fed_model, optimizer_fed, agg_grads)
            elif aggregation == 'bulyan':
                agg_grads, krum_candidate = bulyan(malicious_grads, n_attacker)
                update_fed_model(fed_model, optimizer_fed, agg_grads)
            elif aggregation == 'mkrum':
                agg_grads = multi_krum(malicious_grads, n_attacker)
                update_fed_model(fed_model, optimizer_fed, agg_grads)
            elif aggregation == 'fltrust':
                agg_grads = fl_trust(fed_model, criterion, optimizer_fed, fltrust_data, fltrust_label, malicious_grads)
                update_fed_model(fed_model, optimizer_fed, agg_grads)
            elif aggregation == 'frl' and epoch_num > 450:
                FRL_Vote(fed_model, cur_user_updates, initial_scores)
            elif aggregation == 'flame':
                flame(fed_model, client_weight)
            elif aggregation == 'crowdguard':
                fed_model = crowdguard(malicious_grads, fed_model, user_tr_data_tensors, user_tr_label_tensors, device)
            else:
                agg_grads = torch.mean(malicious_grads, dim=0).to(device)
                fed_model = apply_gradients(fed_model, agg_grads)
                # update_fed_model(fed_model, optimizer_fed, agg_grads)

            del user_grads

            if aggregation != 'frl' and aggregation != 'flame':
                start_idx = 0
                update_fed_model(fed_model, optimizer_fed, agg_grads)
            elif aggregation == 'frl' and epoch_num <= 450:
                start_idx = 0
                update_fed_model(fed_model, optimizer_fed, agg_grads)

            val_loss, val_acc = test(val_data_tensor, val_label_tensor, fed_model, criterion, use_cuda)
            te_loss, te_acc = test(te_data_tensor, te_label_tensor, fed_model, criterion, use_cuda)

            is_best = best_global_acc < val_acc
            best_global_acc = max(best_global_acc, val_acc)
            if is_best:
                best_global_te_acc = te_acc

            if epoch_num % interval == 0 or epoch_num == nepochs - 1:
                print('%s: at %s n_at %d e %d fed_model val loss %.4f val acc %.4f best val_acc %f te_acc %f' % (
                    aggregation, at_type, n_attacker, epoch_num, val_loss, val_acc, best_global_acc, best_global_te_acc))
                acc_history.append(val_acc)

            if val_loss > 10:
                print('val loss %f too high' % val_loss)
                break

            epoch_num += 1

    #     horizontal_results = pd.DataFrame(horizontal_results_list)
    #     vertical_results = pd.DataFrame(vertical_results_list)
    #     weights_results = pd.DataFrame(weights_results_list)
    #     scores_results = pd.DataFrame(scores_results_list)
    #
    #     file_suffix = f"{dataset}_{aggregation}_{at_type}_{detection}_{n_attackers[0]}"
    #     horizontal_results.to_csv(f'result/horizontal_detection_results_{file_suffix}.csv', index=False)
    #     vertical_results.to_csv(f'result/vertical_detection_results_{file_suffix}.csv', index=False)
    #     weights_results.to_csv(f'result/weights_results_{file_suffix}.csv', index=False)
    #     scores_results.to_csv(f'result/scores_results_{file_suffix}.csv', index=False)
    #
    # # Save scores and weights for each client
    # for client_id, scores in client_scores_dict.items():
    #     df = pd.DataFrame(scores)
    #     df.to_csv(f'result/client_{client_id}_scores.csv', index=False)
    #
    # for client_id, weights in client_weights_dict.items():
    #     df = pd.DataFrame(weights)
    #     df.to_csv(f'result/client_{client_id}_weights.csv', index=False)

    file_suffix = f"{dataset}_{aggregation}_{at_type}_{detection}_{n_attackers[0]}"
    #make directory for result
    if not os.path.exists(file_suffix):
        os.makedirs(file_suffix)
    # Save scores and weights for each client
    for client_id, scores in client_scores_dict.items():
        df = pd.DataFrame(scores)
        df.to_csv(f'{file_suffix}/client_{client_id}_scores.csv', index=False)

    for client_id, weights in client_weights_dict.items():
        df = pd.DataFrame(weights)
        df.to_csv(f'{file_suffix}/client_{client_id}_weights.csv', index=False)

    df = pd.DataFrame(server_scores)
    df.to_csv(f'{file_suffix}/server_scores.csv', index=False)

    df = pd.DataFrame(server_weights)
    df.to_csv(f'{file_suffix}/server_weights.csv', index=False)

if __name__ == "__main__":
    cProfile.run('main()', 'profiling_results')