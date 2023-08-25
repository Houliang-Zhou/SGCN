import time

import torch.nn as nn
import numpy as np
import torch
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

# from dataloader import DataLoader  # replace with custom dataloader to handle subgraphs
from imbalanced_snps import ImbalancedDatasetSampler
from torch.utils.data import DataLoader
from util.convert_to_gpu import gpu
from util.convert_to_gpu_and_tensor import gpu_t
from util.convert_to_gpu_scalar import gpu_ts
from util.convert_to_cpu import cpu
from snps_graph import SnpsDataset, parse_go_json
from kernel.go_model import *
from kernel.mlp import *
def cross_validation_with_val_set(dataset,
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
                                  json_path=None):

    final_train_losses, val_losses, accs, durations = [], [], [], []
    for fold, (train_idx, test_idx, val_idx) in enumerate(
            zip(*k_fold(dataset, folds))):

        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]
        val_dataset = dataset[val_idx]

        if 'adj' in train_dataset[0]:
            train_loader = DenseLoader(train_dataset, batch_size, shuffle=True)
            val_loader = DenseLoader(val_dataset, batch_size, shuffle=False)
            test_loader = DenseLoader(test_dataset, batch_size, shuffle=False)
        else:
            train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

        model.to(device).reset_parameters()
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        # if torch.cuda.is_available():
        #     torch.cuda.synchronize()

        t_start = time.perf_counter()

        pbar = tqdm(range(1, epochs + 1), ncols=70)
        cur_val_losses = []
        cur_accs = []
        for epoch in pbar:
            train_loss = train(model, optimizer, train_loader, device)
            cur_val_losses.append(eval_loss(model, val_loader, device))
            cur_accs.append(eval_acc(model, test_loader, device))
            eval_info = {
                'fold': fold,
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': cur_val_losses[-1],
                'test_acc': cur_accs[-1],
            }
            log = 'Fold: %d, train_loss: %0.4f, val_loss: %0.4f, test_acc: %0.4f' % (
                fold, eval_info["train_loss"], eval_info["val_loss"], eval_info["test_acc"]
            )
            pbar.set_description(log)

            if epoch % lr_decay_step_size == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_decay_factor * param_group['lr']

        val_losses += cur_val_losses
        accs += cur_accs

        loss, argmin = tensor(cur_val_losses).min(dim=0)
        acc = cur_accs[argmin.item()]
        final_train_losses.append(eval_info["train_loss"])
        log = 'Fold: %d, final train_loss: %0.4f, best val_loss: %0.4f, test_acc: %0.4f' % (
            fold, eval_info["train_loss"], loss, acc
        )
        print(log)
        if logger is not None:
            logger(log)

        # if torch.cuda.is_available():
        #     torch.cuda.synchronize()

        t_end = time.perf_counter()
        durations.append(t_end - t_start)

    loss, acc, duration = tensor(val_losses), tensor(accs), tensor(durations)
    loss, acc = loss.view(folds, epochs), acc.view(folds, epochs)
    loss, argmin = loss.min(dim=1)
    acc = acc[torch.arange(folds, dtype=torch.long), argmin]
    #average_train_loss = float(np.mean(final_train_losses))
    #std_train_loss = float(np.std(final_train_losses))

    log = 'Val Loss: {:.4f}, Test Accuracy: {:.3f} ± {:.3f}, Duration: {:.3f}'.format(
        loss.mean().item(),
        acc.mean().item(),
        acc.std().item(),
        duration.mean().item()
    ) #+ ', Avg Train Loss: {:.4f}'.format(average_train_loss)
    print(log)
    if logger is not None:
        logger(log)

    return loss.mean().item(), acc.mean().item(), acc.std().item()


def cross_validation_without_val_set( dataset,
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
                                      json_path=None,
                                      disease_id=0):
    go_snps, adj, pool_dim, n_l, go_level, new_go_ids, go_ids_genes_list = parse_go_json(json_path)
    A = gpu(torch.tensor(adj).float().t().to_sparse().coalesce())
    A_g = gpu(torch.tensor(go_snps).float().to_sparse().coalesce())
    pool_dim = np.asarray(pool_dim).tolist()
    l_dim = 16

    score_result = []
    test_losses, accs, durations = [], [], []
    count = 1
    for fold, (train_idx, test_idx, val_idx) in enumerate(
            zip(*k_fold(dataset, folds))):
        print("CV fold " + str(count))
        count += 1

        train_idx = torch.cat([train_idx, val_idx], 0)  # combine train and val
        train_dataset = SnpsDataset(disease_id, train_idx.numpy(),test_idx.numpy(),isTrain=True)
        test_dataset = SnpsDataset(disease_id, train_idx.numpy(),test_idx.numpy(),isTrain=False)

        train_loader = DataLoader(train_dataset, batch_size, shuffle=False, sampler=ImbalancedDatasetSampler(train_dataset))#True  False, sampler=ImbalancedDatasetSampler(train_dataset)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

        model = MLP_Model()
        model.to(device)
        optimizer = Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.7)

        criterion_class = nn.BCELoss(reduction='none')
        criterion_recon = nn.MSELoss(reduction='none')
        lambda0 = gpu_ts(0.00001)
        temperature = gpu_ts(0.1)

        # if torch.cuda.is_available():
        #     torch.cuda.synchronize()
        score_result_epoch = []
        t_start = time.perf_counter()

        pbar = tqdm(range(1, epochs + 1), ncols=70)
        for epoch in pbar:
            train_loss = train(model, optimizer, train_loader, device, criterion_class, criterion_recon, temperature, lambda0)
            test_loss = eval_loss(model, test_loader, device, criterion_class, criterion_recon, temperature, lambda0)
            test_losses.append(test_loss)
            accs.append(eval_acc(model, test_loader, device, criterion_class, criterion_recon, temperature, lambda0))
            true_label, pred_label, acuracy, auc, test_f1, sensitivity, specificity = eval_scores(model, test_loader, device, criterion_class, criterion_recon, temperature, lambda0)
            eval_info = {
                'fold': fold,
                'epoch': epoch,
                'train_loss': train_loss,
                'test_loss': test_losses[-1],
                'test_acc': accs[-1],
                'test_auc': auc,
                'test_f1': test_f1,
                'test_sen': sensitivity,
                'test_spe': specificity,
            }
            log = 'Fold: %d, train_loss: %0.4f, test_loss: %0.4f, test_acc: %0.4f' % (
                fold, eval_info["train_loss"], eval_info["test_loss"], eval_info["test_acc"]
            )
            print_log = 'Fold: %d, epoch:%d, train_loss: %0.4f, test_loss: %0.4f, test_acc: %0.4f, test_auc: %0.4f, test_f1: %0.4f, test_sen: %0.4f, test_spe: %0.4f' % (
                fold, eval_info["epoch"], eval_info["train_loss"], eval_info["test_loss"], eval_info["test_acc"], eval_info["test_auc"], eval_info["test_f1"]
                , eval_info["test_sen"], eval_info["test_spe"]
            )
            # pbar.set_description(log)
            print(print_log)
            if logger is not None:
                logger(print_log)

            if epoch % lr_decay_step_size == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_decay_factor * param_group['lr']
            score_result_epoch.append([eval_info["test_acc"], eval_info["test_auc"], eval_info["test_f1"], eval_info["test_sen"], eval_info["test_spe"]])

            scheduler.step()

        if logger is not None:
            logger(log)
        score_result.append(score_result_epoch)
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
    if result_path is not None:
        with open(result_path, 'wb') as f:
            np.save(f, score_result)

    #return loss.mean().item(), acc_final.item(), acc[:, -1].std().item()
    return loss.mean().item(), acc_max.item(), acc[:, argmax].std().item()


def k_fold(dataset, folds):
    skf = StratifiedKFold(folds, shuffle=True, random_state=1000)

    test_indices, train_indices = [], []
    for _, idx in skf.split(torch.zeros(len(dataset)), dataset.get_labels()):
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
    kf = KFold(folds, shuffle=True, random_state=1000)

    test_indices, train_indices = [], []
    for _, test_idx in kf.split(dataset):
        test_indices.append(torch.from_numpy(test_idx))

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


def train(model, optimizer, loader, device, criterion_class, criterion_recon, temperature, lambda0):
    model.train()

    losses = []
    num_sample = 0
    for data, label in loader:
        num_sample += data.shape[0]
        optimizer.zero_grad()
        data = data.to(device)
        label = label.to(device)

        for param in model.parameters():
            param.requires_grad = True

        y_hat= model(data)
        class_loss = torch.sum(criterion_class(y_hat.view(-1), label.view(-1)))  # classification loss

        loss = class_loss
        loss.backward()

        losses.append(cpu(loss.detach()).data.numpy())

        optimizer.step()
    ll = np.sum(losses) / num_sample

    return ll


def eval_acc(model, loader, device, criterion_class, criterion_recon, temperature, lambda0):
    model.eval()
    num_sample = 0
    correct = 0
    for data, label in loader:
        num_sample += data.shape[0]
        data = data.to(device)
        label = label.to(device)
        with torch.no_grad():
            y_hat = model(data)
        y_te = label>0.5
        y_pred = y_hat>0.5
        correct += y_te.eq(y_pred).sum().cpu().item()
        a=1
    return correct / num_sample


def eval_loss(model, loader, device, criterion_class, criterion_recon, temperature, lambda0):
    model.eval()
    num_sample = 0
    losses = []

    loss = 0
    for data, label in loader:
        num_sample += data.shape[0]
        data = data.to(device)
        label = label.to(device)
        with torch.no_grad():
            y_hat = model(data)
        class_loss = torch.sum(criterion_class(y_hat, label))  # classification loss
        loss = class_loss

        losses.append(cpu(loss.detach()).data.numpy())
    ll = np.sum(losses) / num_sample

    return ll

def eval_scores(model, loader, device, criterion_class, criterion_recon, temperature, lambda0):
    model.eval()
    true_label = []
    pred_label = []
    all_scores = []
    correct = 0
    num_sample = 0
    for data, label in loader:
        num_sample += data.shape[0]
        data = data.to(device)
        label = label.to(device)
        with torch.no_grad():
            y_hat= model(data)
            all_scores.append(y_hat.view(-1).cpu().detach())
        pred = y_hat > 0.5
        y_te = label > 0.5
        correct += pred.eq(y_te).sum().cpu().item()
        true_label = true_label + [per_label for per_label in y_te.view(-1).cpu().numpy().tolist()]
        pred_label = pred_label + [per_label for per_label in pred.view(-1).cpu().numpy().tolist()]
        a=1
    all_scores = torch.cat(all_scores).cpu().numpy()
    auc = 0
    try:
        fpr, tpr, _ = metrics.roc_curve(np.asarray(true_label), all_scores, pos_label=1)
        auc = metrics.auc(fpr, tpr)
    except:
        auc = 0
    true_label = np.asarray(true_label)
    pred_label = np.asarray(pred_label)
    test_f1 = f1_score(true_label, pred_label, average='weighted')
    cm = confusion_matrix(true_label, pred_label)
    TN, FP, FN, TP = cm.ravel()
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    return true_label, pred_label, correct / num_sample, auc, test_f1, sensitivity, specificity