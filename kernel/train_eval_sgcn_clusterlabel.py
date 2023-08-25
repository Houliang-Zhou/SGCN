import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor
from torch.optim import Adam
from sklearn.model_selection import StratifiedKFold, KFold
from torch_geometric.data import DenseDataLoader as DenseLoader
from tqdm import tqdm
import pdb
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import statistics
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

from dataloader import DataLoader  # replace with custom dataloader to handle subgraphs
# from torch_geometric.loader import DataLoader
from imbalanced import ImbalancedDatasetSampler
import sgcn_hyperparameters as hp

from util.convert_to_gpu import gpu
from util.convert_to_gpu_and_tensor import gpu_t
from util.convert_to_gpu_scalar import gpu_ts
from util.convert_to_cpu import cpu
from snps_graph import SnpsDataset, parse_go_json
from kernel.sgcn_img_snp_clusterlabel import SGCN_GCN_CLUSTERLABEL


def cross_validation_with_val_set(dataset,
                                  num_layers,
                                  hidden,
                                  folds,
                                  epochs,
                                  batch_size,
                                  lr,
                                  lr_decay_factor,
                                  lr_decay_step_size,
                                  weight_decay,
                                  device,
                                  logger=None,
                                  result_path=None):
    pass
    # final_train_losses, val_losses, accs, durations = [], [], [], []
    # for fold, (train_idx, test_idx, val_idx) in enumerate(
    #         zip(*k_fold(dataset, folds))):
    #
    #     train_dataset = dataset[train_idx]
    #     test_dataset = dataset[test_idx]
    #     val_dataset = dataset[val_idx]
    #
    #     if 'adj' in train_dataset[0]:
    #         train_loader = DenseLoader(train_dataset, batch_size, shuffle=True)
    #         val_loader = DenseLoader(val_dataset, batch_size, shuffle=False)
    #         test_loader = DenseLoader(test_dataset, batch_size, shuffle=False)
    #     else:
    #         train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    #         val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
    #         test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    #
    #     model.to(device).reset_parameters()
    #     model = model.to(device)
    #     optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    #
    #     # if torch.cuda.is_available():
    #     #     torch.cuda.synchronize()
    #
    #     t_start = time.perf_counter()
    #
    #     pbar = tqdm(range(1, epochs + 1), ncols=70)
    #     cur_val_losses = []
    #     cur_accs = []
    #     for epoch in pbar:
    #         train_loss = train(model, optimizer, train_loader, device)
    #         cur_val_losses.append(eval_loss(model, val_loader, device))
    #         cur_accs.append(eval_acc(model, test_loader, device))
    #         eval_info = {
    #             'fold': fold,
    #             'epoch': epoch,
    #             'train_loss': train_loss,
    #             'val_loss': cur_val_losses[-1],
    #             'test_acc': cur_accs[-1],
    #         }
    #         log = 'Fold: %d, train_loss: %0.4f, val_loss: %0.4f, test_acc: %0.4f' % (
    #             fold, eval_info["train_loss"], eval_info["val_loss"], eval_info["test_acc"]
    #         )
    #         pbar.set_description(log)
    #
    #         if epoch % lr_decay_step_size == 0:
    #             for param_group in optimizer.param_groups:
    #                 param_group['lr'] = lr_decay_factor * param_group['lr']
    #
    #     val_losses += cur_val_losses
    #     accs += cur_accs
    #
    #     loss, argmin = tensor(cur_val_losses).min(dim=0)
    #     acc = cur_accs[argmin.item()]
    #     final_train_losses.append(eval_info["train_loss"])
    #     log = 'Fold: %d, final train_loss: %0.4f, best val_loss: %0.4f, test_acc: %0.4f' % (
    #         fold, eval_info["train_loss"], loss, acc
    #     )
    #     print(log)
    #     if logger is not None:
    #         logger(log)
    #
    #     # if torch.cuda.is_available():
    #     #     torch.cuda.synchronize()
    #
    #     t_end = time.perf_counter()
    #     durations.append(t_end - t_start)
    #
    # loss, acc, duration = tensor(val_losses), tensor(accs), tensor(durations)
    # loss, acc = loss.view(folds, epochs), acc.view(folds, epochs)
    # loss, argmin = loss.min(dim=1)
    # acc = acc[torch.arange(folds, dtype=torch.long), argmin]
    # #average_train_loss = float(np.mean(final_train_losses))
    # #std_train_loss = float(np.std(final_train_losses))
    #
    # log = 'Val Loss: {:.4f}, Test Accuracy: {:.3f} ± {:.3f}, Duration: {:.3f}'.format(
    #     loss.mean().item(),
    #     acc.mean().item(),
    #     acc.std().item(),
    #     duration.mean().item()
    # ) #+ ', Avg Train Loss: {:.4f}'.format(average_train_loss)
    # print(log)
    # if logger is not None:
    #     logger(log)
    #
    # return loss.mean().item(), acc.mean().item(), acc.std().item()


def cross_validation_without_val_set( dataset,
                                      num_layers,
                                      hidden,
                                      folds,
                                      epochs,
                                      batch_size,
                                      lr,
                                      lr_decay_factor,
                                      lr_decay_step_size,
                                      weight_decay,
                                      device,
                                      logger=None,
                                      result_path=None,
                                      result_path_classify = None,
                                      json_path=None,
                                      disease_id=0,
                                      isCrossAtten=False,
                                      isPredictCluster = True
                                      ):
    go_snps, adj, pool_dim, n_l, go_level, new_go_ids, go_ids_genes_list = parse_go_json(json_path)
    A = gpu(torch.tensor(adj).float().t().to_sparse().coalesce(), device)
    A_g = gpu(torch.tensor(go_snps).float().to_sparse().coalesce(), device)
    pool_dim = np.asarray(pool_dim).tolist()
    l_dim = 32

    score_result = []
    score_result_classify = []
    test_losses, accs, durations = [], [], []
    count = 1
    for fold, (train_idx, test_idx, val_idx) in enumerate(
            zip(*k_fold(dataset, folds))):
        print("CV fold " + str(count))
        count += 1

        train_idx = torch.cat([train_idx, val_idx], 0)  # combine train and val
        # test_idx = torch.cat([test_idx, val_idx], 0)
        # train_idx = torch.cat([train_idx], 0)
        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]
        # train_dataset = [dataset[i] for i in train_idx]
        # test_dataset = [dataset[i] for i in test_idx]

        # if 'adj' in train_dataset[0]:
        #     train_loader = DenseLoader(train_dataset, batch_size, shuffle=False, sampler=ImbalancedDatasetSampler(train_dataset))#True  False, sampler=ImbalancedDatasetSampler(train_dataset)
        #     test_loader = DenseLoader(test_dataset, batch_size, shuffle=False)
        # else:
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)#True  False, sampler=ImbalancedDatasetSampler(train_dataset)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

        model = SGCN_GCN_CLUSTERLABEL(num_layers, hidden, A_g, A, pool_dim, l_dim, device, isCrossAtten=isCrossAtten, isPredictCluster=isPredictCluster)
        model = model.to(device)
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        criterion_recon = nn.MSELoss(reduction='none')
        lambda0 = gpu_ts(0.00001, device) #0.00001
        temperature = gpu_ts(0.1, device)

        lambda1 = gpu_ts(0.0, device) #0.0

        # if torch.cuda.is_available():
        #     torch.cuda.synchronize()
        score_result_epoch = []
        score_result_epoch_classify = []
        t_start = time.perf_counter()

        pbar = tqdm(range(1, epochs + 1), ncols=70)
        for epoch in pbar:
            train_loss = train(model, optimizer, train_loader, temperature, lambda0, lambda1, criterion_recon, device, isPredictCluster)
            test_losses.append(eval_loss(model, test_loader, temperature, lambda0, lambda1, criterion_recon,device, isPredictCluster))
            acc_classify, acc_cluster = eval_acc(model, test_loader, temperature, lambda0, lambda1, criterion_recon,device)

            classify_result, clus_result = eval_scores(model, test_loader, temperature, lambda0, lambda1, criterion_recon,device)
            true_label_classify, pred_label_classify, acuracy_classify, auc_classify, test_f1_classify, sensitivity_classify, specificity_classify, precision_classify, result4eachclass, cm_clas = classify_result
            true_label, pred_label, acuracy, auc, test_f1, sensitivity, specificity, precision = clus_result
            accs.append(acuracy)

            eval_info = {
                'fold': fold,
                'epoch': epoch,
                'train_loss': train_loss,
                'test_loss': test_losses[-1],
                'test_acc': accs[-1],
                'test_auc': auc,
                'test_f1': test_f1,
                'test_sen': sensitivity,
                'test_pre': precision,
            }
            log = 'Fold: %d, train_loss: %0.4f, test_loss: %0.4f, test_acc: %0.4f' % (
                fold, eval_info["train_loss"], eval_info["test_loss"], eval_info["test_acc"]
            )
            print_log = 'Cluster: Fold: %d, epoch:%d, train_loss: %0.4f, test_loss: %0.4f, test_acc: %0.4f, test_auc: %0.4f, test_f1: %0.4f, test_sen: %0.4f, test_pre: %0.4f' % (
                fold, eval_info["epoch"], eval_info["train_loss"], eval_info["test_loss"], eval_info["test_acc"], eval_info["test_auc"], eval_info["test_f1"]
                , eval_info["test_sen"], eval_info["test_pre"]
            )
            print_log_Classification = 'Classification: Fold: %d, epoch:%d, train_loss: %0.4f, test_loss: %0.4f, test_acc: %0.4f, test_auc: %0.4f, test_f1: %0.4f, test_sen: %0.4f, test_pre: %0.4f' % (
                fold, eval_info["epoch"], eval_info["train_loss"], eval_info["test_loss"], acuracy_classify,
                auc_classify, test_f1_classify, sensitivity_classify, precision_classify
            )
            # pbar.set_description(log)
            print(print_log)
            print(print_log_Classification)
            print(cm_clas)
            if logger is not None:
                logger(print_log)
                logger(print_log_Classification)
                logger(cm_clas)

            if epoch % lr_decay_step_size == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_decay_factor * param_group['lr']
            score_result_epoch.append([eval_info["test_acc"], eval_info["test_auc"], eval_info["test_f1"], eval_info["test_sen"], eval_info["test_pre"]])
            save_result_classify = []
            save_result_classify += [acuracy_classify, auc_classify, test_f1_classify, sensitivity_classify, precision_classify]
            for item_each_class in result4eachclass:
                save_result_classify += [item_each_class]
            score_result_epoch_classify.append(save_result_classify)

        if logger is not None:
            logger(log)
        score_result.append(score_result_epoch)
        score_result_classify.append(score_result_epoch_classify)
        # if torch.cuda.is_available():
        #     torch.cuda.synchronize()

        t_end = time.perf_counter()
        durations.append(t_end - t_start)

    loss, acc, duration = tensor(test_losses), tensor(accs), tensor(durations)
    loss, acc = loss.view(folds, epochs), acc.view(folds, epochs)
    acc_mean = acc.mean(0)
    acc_max, argmax = acc_mean.max(dim=0)
    acc_final = acc_mean[-1]

    log = ('Test Loss: {:.4f}, Test Max Accuracy: {:.3f} ± {:.3f}, ' +
          'Test Final Accuracy: {:.3f} ± {:.3f}, Duration: {:.3f}').format(
        loss.mean().item(),
        acc_max.item(),
        acc[:, argmax].std().item(),
        acc_final.item(),
        acc[:, -1].std().item(),
        duration.mean().item()
    )
    print(log)
    if logger is not None:
        logger(log)

    score_result = np.asarray(score_result)
    score_result_classify = np.asarray(score_result_classify)
    if result_path is not None:
        with open(result_path, 'wb') as f:
            np.save(f, score_result)
    if result_path_classify is not None:
        with open(result_path_classify, 'wb') as f:
            np.save(f, score_result_classify)

    #return loss.mean().item(), acc_final.item(), acc[:, -1].std().item()
    return loss.mean().item(), acc_max.item(), acc[:, argmax].std().item()


def k_fold(dataset, folds):
    skf = StratifiedKFold(folds, shuffle=True, random_state=1000) #1000

    test_indices, train_indices = [], []
    # tmp_label = [dataset[index].y.item() for index in range(len(dataset))]
    for _, idx in skf.split(torch.zeros(len(dataset)), dataset.data.y[dataset.indices()]):  #
        test_indices.append(torch.from_numpy(idx).long())

    val_indices = [test_indices[i - 1] for i in range(folds)]

    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.uint8)
        train_mask[test_indices[i]] = 0
        # try:
        #     train_mask[test_indices[i]] = 0
        # except:
        #     a=1
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero().view(-1))

    return train_indices, test_indices, val_indices


def k_fold2(dataset, folds):
    kf = KFold(folds, shuffle=True, random_state=1000) #1000

    test_indices, train_indices = [], []
    for _, test_idx in kf.split(dataset):
        test_indices.append(torch.from_numpy(test_idx).long())

    val_indices = [test_indices[i - 1] for i in range(folds)]

    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.uint8)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero().view(-1))

    return train_indices, test_indices, val_indices


def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)

def get_classify_report(y_true, y_pred):
    result = classification_report(y_true, y_pred)  # classification report's output is a string
    lines = result.split('\n')  # extract every line and store in a list
    res = []  # list to store the cleaned results
    for i in range(len(lines)):
        line = lines[i].split(" ")  # Values are separated by blanks. Split at the blank spaces.
        line = [j for j in line if j != '']  # add only the values into the list
        if len(line) != 0:
            # empty lines get added as empty lists. Remove those
            res.append(line)

    acc = float(res[4][1])
    weig_precision = float(res[-1][2])
    weig_recall = float(res[-1][3])
    weig_f1 = float(res[-1][4])

    result4eachclass = []
    for i in range(1,4):
        for j in range(1, 4):
            result4eachclass.append(float(res[i][j]))

    # print(acc, weig_precision, weig_recall, weig_f1)
    return acc, weig_precision, weig_recall, weig_f1, result4eachclass


def train(model, optimizer, loader, temperature, lambda0, lambda1, criterion_recon, device, isPredictCluster, num_cluster=2):
    model.train()

    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        data = data.to(device)
        for param in model.parameters():
            param.requires_grad = True

        out, out_cluster, snps_hat, out_feat = model(data, temperature, device)
        loss_ce = F.nll_loss(out, data.y.view(-1))
        loss_ce_cluster = F.nll_loss(out_cluster, data.clust_y.view(-1))

        out_prob, out_cluster_prob, snps_hat_prob, out_feat_prob = model(data, temperature, device, isExplain=True)
        loss_mi = F.nll_loss(out_prob, data.y.view(-1))
        loss_mi_cluster = F.nll_loss(out_cluster_prob, data.clust_y.view(-1))

        loss_prob = model.loss_probability(data.x, data.edge_index, data.edge_attr, hp)

        recon_loss = (lambda0 * torch.sum(criterion_recon(snps_hat, data.snps_feat)) + lambda0 * torch.sum(criterion_recon(snps_hat_prob, data.snps_feat)))/2
        cluster_loss = 0
        for c in range(num_cluster):
            cluster_loss += lambda1*(model.consist_loss(out_feat[data.clust_y == c]) + model.consist_loss(out_feat_prob[data.clust_y == c]))/2

        if isPredictCluster:
            loss = hp.lamda_ce * (loss_ce+loss_ce_cluster)/2 + hp.lamda_mi * (loss_mi+loss_mi_cluster)/2 + loss_prob + recon_loss
        else:
            loss = hp.lamda_ce * (loss_ce) + hp.lamda_mi * (loss_mi) + loss_prob + recon_loss

        # loss = F.nll_loss(out, data.y.view(-1))
        loss.backward()
        total_loss += loss.detach().cpu().item() * num_graphs(data)
        optimizer.step()
    return total_loss / len(loader.dataset)


def eval_loss(model, loader, temperature, lambda0, lambda1, criterion_recon,device, isPredictCluster,num_cluster=2):
    model.eval()

    loss = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out, out_cluster, snps_hat, out_feat = model(data, temperature, device)
            out_prob, out_cluster_prob, snps_hat_prob, out_feat_prob = model(data, temperature, device, isExplain=True)

        # loss += F.nll_loss(out, data.y.view(-1), reduction='sum').detach().cpu().item()
        loss_ce = F.nll_loss(out, data.y.view(-1))  # , weight=weight
        loss_ce_cluster = F.nll_loss(out_cluster, data.clust_y.view(-1))
        loss_mi = F.nll_loss(out_prob, data.y.view(-1))
        loss_mi_cluster = F.nll_loss(out_cluster_prob, data.clust_y.view(-1))
        recon_loss = lambda0 * torch.sum(criterion_recon(snps_hat, data.snps_feat)) + lambda0 * torch.sum(
            criterion_recon(snps_hat_prob, data.snps_feat))
        cluster_loss = 0
        for c in range(num_cluster):
            cluster_loss += lambda1 * (model.consist_loss(out_feat[data.clust_y == c]) + model.consist_loss(
                out_feat_prob[data.clust_y == c])) / 2
        # all_scores.append(out_prob[:,1].cpu().detach())
        # pred_y = out_prob.data.max(1, keepdim=True)[1]
        loss_prob = model.loss_probability(data.x, data.edge_index, data.edge_attr, hp)
        if isPredictCluster:
            loss += (hp.lamda_ce * (loss_ce+loss_ce_cluster)/2 + hp.lamda_mi * (loss_mi+loss_mi_cluster)/2 + loss_prob + recon_loss).cpu().item() * num_graphs(data)
        else:
            loss += (hp.lamda_ce * (loss_ce) + hp.lamda_mi * (loss_mi) + loss_prob + recon_loss).cpu().item() * num_graphs(
                data)

    return loss / len(loader.dataset)

def eval_acc(model, loader, temperature, lambda0, lambda1, criterion_recon,device):
    model.eval()

    correct = 0
    correct_clus = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out, out_cluster, _, _ = model(data, temperature, device)
            pred = out.max(1)[1]
            pred_clus = out_cluster.max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
        correct_clus += pred_clus.eq(data.clust_y.view(-1)).sum().item()
    return correct / len(loader.dataset), correct_clus / len(loader.dataset)

def eval_scores(model, loader, temperature, lambda0, lambda1, criterion_recon,device):
    model.eval()
    true_label_clas = []
    pred_label_clas = []
    true_label_clust = []
    pred_label_clust = []
    all_scores = []
    correct = 0
    correct_clus = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out, out_cluster, _, _ = model(data, temperature, device)
            all_scores.append(out_cluster[:, 1].cpu().detach())
            pred = out.max(1)[1]
            pred_clus = out_cluster.max(1)[1]

        correct += pred.eq(data.y.view(-1)).sum().item()
        correct_clus += pred_clus.eq(data.clust_y.view(-1)).sum().item()

        true_label_clas = true_label_clas + [per_label for per_label in data.y.cpu().numpy().tolist()]
        pred_label_clas = pred_label_clas + [per_label for per_label in pred.cpu().numpy().tolist()]

        true_label_clust = true_label_clust + [per_label for per_label in data.clust_y.cpu().numpy().tolist()]
        pred_label_clust = pred_label_clust + [per_label for per_label in pred_clus.cpu().numpy().tolist()]

    all_scores = torch.cat(all_scores).cpu().numpy()
    auc = 0
    try:
        fpr, tpr, _ = metrics.roc_curve(np.asarray(true_label_clust), all_scores, pos_label=1)
        auc = metrics.auc(fpr, tpr)
    except:
        auc = 0
    true_label_clust = np.asarray(true_label_clust)
    pred_label_clust = np.asarray(pred_label_clust)
    test_f1 = f1_score(true_label_clust, pred_label_clust, average='weighted')
    cm = confusion_matrix(true_label_clust, pred_label_clust)
    TN, FP, FN, TP = cm.ravel()
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    precision = TP / (TP + FP)
    clus_result = (true_label_clust, pred_label_clust, correct_clus / len(loader.dataset), auc, test_f1, sensitivity, specificity, precision)

    true_label_clas = np.asarray(true_label_clas)
    pred_label_clas = np.asarray(pred_label_clas)
    cm_clas = confusion_matrix(true_label_clas, pred_label_clas)
    acc, weig_precision, weig_recall, weig_f1, result4eachclass = get_classify_report(true_label_clas, pred_label_clas)
    classify_result = (true_label_clas, pred_label_clas, acc, 0, weig_f1, weig_recall, 0, weig_precision, result4eachclass, cm_clas)

    return classify_result, clus_result