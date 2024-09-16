import time
import os
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

from dataloader import DataLoader  # replace with custom dataloader to handle subgraphs
# from torch_geometric.loader import DataLoader
from imbalanced import ImbalancedDatasetSampler
from sgcn_data import *

def cross_validation_with_val_set(args,dataset,
                                      model,
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
                                      pre_transform=None,
                                      result_file_name=None):
    test_data = []
    if args.isTestAdnitype:
        train_data, test_data = separate_data_adnitype(dataset, disease_id=args.disease_id4Adnitype,
                                                       adnitype_id=args.adnitype_id)
        train_data = ADNIDataset('./', 'SGCN_train', train_data, pre_transform=pre_transform)
        test_data = ADNIDataset('./', 'SGCN_test', test_data, pre_transform=pre_transform)
        dataset = train_data
    score_result = []
    test_losses, accs, durations = [], [], []
    val_losses = []
    count = 1
    for fold, (train_idx, test_idx, val_idx) in enumerate(
            zip(*k_fold(dataset, folds))):
        print("CV fold " + str(count))
        count += 1

        train_idx = torch.cat([train_idx], 0)  # combine train and val
        train_dataset = dataset[train_idx]
        val_dataset = dataset[val_idx]
        test_dataset = dataset[test_idx]
        train_loader = DataLoader(train_dataset, batch_size, shuffle=False, sampler=ImbalancedDatasetSampler(
            train_dataset))  # True  False, sampler=ImbalancedDatasetSampler(train_dataset)
        val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

        model.to(device).reset_parameters()
        model = model.to(device)
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        score_result_epoch = []
        t_start = time.perf_counter()

        best_val_loss=np.inf
        pbar = tqdm(range(1, epochs + 1), ncols=70)
        for epoch in pbar:
            train_loss = train(model, optimizer, train_loader, device, args)
            test_losses.append(eval_loss(model, test_loader, device, args))
            val_losses.append(eval_loss(model, val_loader, device, args))
            accs.append(eval_acc(model, test_loader, device, args))
            true_label, pred_label, acuracy, auc, test_f1, sensitivity, specificity = eval_scores(model, test_loader,
                                                                                                  device, args)
            eval_info = {
                'fold': fold,
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_losses[-1],
                'test_loss': test_losses[-1],
                'test_acc': accs[-1],
                'test_auc': auc,
                'test_f1': test_f1,
                'test_sen': sensitivity,
                'test_spe': specificity,
            }
            log = 'Fold: %d, train_loss: %0.4f, val_loss: %0.4f, test_loss: %0.4f, test_acc: %0.4f' % (
                fold, eval_info["train_loss"], eval_info["val_loss"], eval_info["test_loss"], eval_info["test_acc"]
            )
            print_log = 'Fold: %d, epoch:%d, train_loss: %0.4f, val_loss: %0.4f, test_loss: %0.4f, test_acc: %0.4f, test_auc: %0.4f, test_f1: %0.4f, test_sen: %0.4f, test_spe: %0.4f' % (
                fold, eval_info["epoch"], eval_info["train_loss"], eval_info["val_loss"], eval_info["test_loss"], eval_info["test_acc"],
                eval_info["test_auc"], eval_info["test_f1"]
                , eval_info["test_sen"], eval_info["test_spe"]
            )
            # pbar.set_description(log)
            print(print_log)
            if logger is not None:
                logger(print_log)

            if epoch % lr_decay_step_size == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_decay_factor * param_group['lr']
            score_result_epoch.append(
                [eval_info["test_acc"], eval_info["test_auc"], eval_info["test_f1"], eval_info["test_sen"],
                 eval_info["test_spe"]])
            if best_val_loss > eval_info["val_loss"]:
                best_val_loss = eval_info["val_loss"]
                model_path = os.path.join(args.res_dir, 'gcn_state_dict_%s_fold_%d.pt' % (result_file_name, fold))
                torch.save(model.state_dict(), model_path)

        if logger is not None:
            logger(log)
        score_result.append(score_result_epoch)
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

    return loss.mean().item(), acc_max.item(), acc[:, argmax].std().item()


def cross_validation_without_val_set( args,dataset,
                                      model,
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
                                      pre_transform=None,
                                      result_file_name=None):
    test_data = []
    if args.isTestAdnitype:
        train_data, test_data = separate_data_adnitype(dataset, disease_id=args.disease_id4Adnitype, adnitype_id=args.adnitype_id)
        train_data = ADNIDataset('./', 'SGCN_train', train_data, pre_transform=pre_transform)
        test_data = ADNIDataset('./', 'SGCN_test', test_data, pre_transform=pre_transform)
        dataset = train_data
    score_result = []
    test_losses, accs, durations = [], [], []
    count = 1
    for fold, (train_idx, test_idx, val_idx) in enumerate(
            zip(*k_fold(dataset, folds))):
        print("CV fold " + str(count))
        count += 1

        train_idx = torch.cat([train_idx, val_idx], 0)  # combine train and val
        train_dataset = dataset[train_idx]
        if args.isShowValResult:
            test_dataset = dataset[test_idx]
        else:
            test_dataset = test_data

        train_loader = DataLoader(train_dataset, batch_size, shuffle=False, sampler=ImbalancedDatasetSampler(train_dataset))#True  False, sampler=ImbalancedDatasetSampler(train_dataset)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

        model.to(device).reset_parameters()
        model = model.to(device)
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        score_result_epoch = []
        t_start = time.perf_counter()

        pbar = tqdm(range(1, epochs + 1), ncols=70)
        for epoch in pbar:
            train_loss = train(model, optimizer, train_loader, device)
            test_losses.append(eval_loss(model, test_loader, device))
            accs.append(eval_acc(model, test_loader, device))
            true_label, pred_label, acuracy, auc, test_f1, sensitivity, specificity = eval_scores(model, test_loader, device)
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

        if logger is not None:
            logger(log)
        score_result.append(score_result_epoch)
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

    return loss.mean().item(), acc_max.item(), acc[:, argmax].std().item()


def k_fold(dataset, folds):
    skf = StratifiedKFold(folds, shuffle=True, random_state=1000)

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


def train(model, optimizer, loader, device, args):
    model.train()

    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data)
        loss_ce = F.nll_loss(out, data.y.view(-1))
        out_prob = model(data, True)
        loss_mi = F.nll_loss(out_prob, data.y.view(-1))
        loss_prob = model.loss_probability(data.x, data.edge_index, data.edge_attr, args)
        loss = args.lamda_ce * loss_ce + loss_prob + args.lamda_mi * loss_mi
        # loss = F.nll_loss(out, data.y.view(-1))
        loss.backward()
        total_loss += loss.detach().cpu().item() * num_graphs(data)
        optimizer.step()
    return total_loss / len(loader.dataset)


def eval_acc(model, loader, device, args):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data, True).max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
    return correct / len(loader.dataset)


def eval_loss(model, loader, device, args):
    model.eval()

    loss = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data)
            out_prob = model(data, True)

        # loss += F.nll_loss(out, data.y.view(-1), reduction='sum').detach().cpu().item()
        loss_ce = F.nll_loss(out, data.y.view(-1))  # , weight=weight
        loss_mi = F.nll_loss(out_prob, data.y.view(-1))
        # all_scores.append(out_prob[:,1].cpu().detach())
        # pred_y = out_prob.data.max(1, keepdim=True)[1]
        loss_prob = model.loss_probability(data.x, data.edge_index, data.edge_attr, args)
        loss += (args.lamda_ce * loss_ce + loss_prob + args.lamda_mi * loss_mi).cpu().item() * num_graphs(data)
    return loss / len(loader.dataset)

def eval_scores(model, loader, device, args):
    model.eval()
    true_label = []
    pred_label = []
    all_scores = []
    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data, True)
            all_scores.append(out[:, 1].cpu().detach())
            pred = out.max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
        true_label = true_label + [per_label for per_label in data.y.cpu().numpy().tolist()]
        pred_label = pred_label + [per_label for per_label in pred.cpu().numpy().tolist()]
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
    return true_label, pred_label, correct / len(loader.dataset), auc, test_f1, sensitivity, specificity