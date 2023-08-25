import time
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor
from torch.optim import Adam
from torch_geometric.utils import to_dense_batch
from sklearn.model_selection import StratifiedKFold, KFold
from torch_geometric.data import DenseDataLoader as DenseLoader
from tqdm import tqdm
import pdb
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import statistics
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr, spearmanr

from dataloader import DataLoader  # replace with custom dataloader to handle subgraphs
# from torch_geometric.loader import DataLoader
from imbalanced import ImbalancedDatasetSampler
import sgcn_hyperparameters as hp

from util.output import output_npy, output_importance
from util.convert_to_gpu import gpu
from util.convert_to_gpu_and_tensor import gpu_t
from util.convert_to_gpu_scalar import gpu_ts
from util.convert_to_cpu import cpu
from snps_graph import SnpsDataset, parse_go_json
from kernel.sgcn_img_snp import SGCN_GCN_IMGSNP
from util.tool import KNNImputation

def cross_validation_with_val_set(dataset,
                                  args,
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
                                  result_file_name=None,
                                  json_path=None,
                                  disease_id=0,
                                  isCrossAtten=False,
                                  isSoftSimilarity=False,
                                  rbf_gamma=0.005,
                                  num_classes=2,
                                  lambda_loss=None,
                                  num_regr=4,
                                  model4eachregr = False,
                                  scaler4score=None
                                  ):
    if lambda_loss is None:
        lambda_loss = [1.0, 1.0, 1.0, 0.0000025, 0.2, 0.2]
    num_rois = 90
    feature_dim = 3
    if args.isMultiFusion:
        num_rois = num_rois * feature_dim
        feature_dim = 1
    go_snps, adj, pool_dim, n_l, go_level, new_go_ids, go_ids_genes_list = parse_go_json(json_path)
    A = gpu(torch.tensor(adj).float().t().to_sparse().coalesce(), device)
    A_g = gpu(torch.tensor(go_snps).float().to_sparse().coalesce(), device)
    pool_dim = np.asarray(pool_dim).tolist()
    l_dim = 32

    test_out_hiddens = []
    test_out_subids = []
    test_out_linear = []
    score_result = []
    test_losses, accs, durations = [], [], []
    count = 1
    best_true_clini_scores = []
    best_true_clini_score_labels = []
    best_pred_clini_scores = []

    for fold, (train_idx, test_idx, val_idx) in enumerate(
            zip(*k_fold(dataset, folds, args))):
        print("CV fold " + str(count))
        count += 1

        train_idx = torch.cat([train_idx], 0)
        train_dataset = dataset[train_idx]
        val_dataset = dataset[val_idx]
        train_dataset, val_dataset = KNNImputation(args, train_dataset, val_dataset, scaler4score)

        train_loader = DataLoader(train_dataset, batch_size,
                                  shuffle=True)  # False, sampler=ImbalancedDatasetSampler(train_dataset)
        val_loader = DataLoader(val_dataset, batch_size, shuffle=False)

        model = SGCN_GCN_IMGSNP(num_layers, hidden, A_g, A, pool_dim, l_dim, device, rois=num_rois, H_0=feature_dim,
                                num_classes=num_classes, isSoftSimilarity=isSoftSimilarity, rbf_gamma=rbf_gamma,
                                isCrossAtten=isCrossAtten, num_regr=num_regr,
                                model4eachregr=model4eachregr, isuseFeat4Regr=args.isuseFeat4Regr,
                                isImageOnly=args.isImageOnly, isSNPsOnly=args.isSNPsOnly,
                                isMultiFusion=args.isMultiFusion)
        model = model.to(device)
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        criterion_recon = nn.MSELoss(reduction='none')
        temperature = gpu_ts(0.1, device)
        score_result_epoch = []
        t_start = time.perf_counter()

        best_loss = np.inf
        best_corr_regr = -np.inf
        best_all_out_hidden = []
        best_all_out_subid = []
        best_all_out_linear = []
        best_true_clini_score = []
        best_true_clini_score_label = []
        best_pred_clini_score = []
        pbar = tqdm(range(1, epochs + 1), ncols=70)
        for epoch in pbar:
            train_loss = train(model, optimizer, train_loader, temperature, lambda_loss, criterion_recon,
                               isSoftSimilarity, device)
            test_losses.append(
                eval_loss(model, val_loader, temperature, lambda_loss, criterion_recon, isSoftSimilarity, device))
            accs.append(
                eval_acc(model, val_loader, temperature, lambda_loss, criterion_recon, isSoftSimilarity, device))
            true_label, pred_label, acuracy, auc, test_f1, sensitivity, specificity, all_out_hidden, all_out_subid, all_out_linear, regression_result = eval_scores(
                model, val_loader, temperature, lambda_loss, criterion_recon, isSoftSimilarity, device,
                num_classes=num_classes, num_regr=num_regr)
            true_clini_score, pred_clini_score, corr, r2, mse = regression_result
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
            print_log = 'Fold: %d, epoch:%d, train_loss: %0.4f, test_loss: %0.4f, test_acc: %0.4f, test_auc: %0.4f, test_f1: %0.4f, test_sen: %0.4f, test_spe: %0.4f,' \
                        ' ' % (
                            fold, eval_info["epoch"], eval_info["train_loss"], eval_info["test_loss"],
                            eval_info["test_acc"], eval_info["test_auc"], eval_info["test_f1"]
                            , eval_info["test_sen"], eval_info["test_spe"]
                        )
            if args.clinical_score_index == -1:
                scores_name = ['tau', 'adas13', 'mmse']  # 'tau',
                for each_regr in range(len(corr)):
                    print_log += '; %s corr: %.5f, r2: %.5f, mse: %.5f' % (
                        scores_name[each_regr], corr[each_regr], r2[each_regr], mse[each_regr])
            else:
                scores_name = ['label', 'age', 'edu', 'sex', 'abeta', 'tau', 'ptau', 'adas13', 'mmse']
                print_log += '; %s corr: %.5f, r2: %.5f, mse: %.5f' % (
                    scores_name[args.clinical_score_index], corr[0], r2[0], mse[0])

            print(print_log)
            if logger is not None:
                logger(print_log)

            if epoch % lr_decay_step_size == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_decay_factor * param_group['lr']
            score_result_epoch.append(
                [eval_info["test_acc"], eval_info["test_auc"], eval_info["test_f1"], eval_info["test_sen"],
                 eval_info["test_spe"]])

            if best_loss > eval_info["test_loss"]:
                best_loss = eval_info["test_loss"]
                best_all_out_hidden = all_out_hidden
                best_all_out_subid = all_out_subid
                best_all_out_linear = all_out_linear
                best_true_clini_score = true_clini_score
                best_true_clini_score_label = true_label
                best_pred_clini_score = pred_clini_score
                model_path = os.path.join(args.res_dir, 'gcn_state_dict_%s_fold_%d.pt' % (result_file_name, fold))
                torch.save(model.state_dict(), model_path)

            if args.isPermutTest:
                best_true_clini_score = true_clini_score
                best_true_clini_score_label = true_label
                best_pred_clini_score = pred_clini_score

        if logger is not None:
            logger(log)
        score_result.append(score_result_epoch)

        output_importance(args, result_file_name, model, fold)

        test_out_hiddens.append(best_all_out_hidden)
        test_out_subids.append(best_all_out_subid)
        test_out_linear.append(best_all_out_linear)

        best_true_clini_scores.append(best_true_clini_score)
        best_true_clini_score_labels.append(best_true_clini_score_label)
        best_pred_clini_scores.append(best_pred_clini_score)

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
    output_npy(result_path, score_result, args=args)

    test_out_hiddens = torch.cat(test_out_hiddens).numpy()
    test_out_subids = torch.cat(test_out_subids).numpy()
    test_out_linear = torch.cat(test_out_linear).numpy()
    hidden_path = os.path.join(args.res_dir, 'hidden_%s.npy' % (result_file_name))
    subids_path = os.path.join(args.res_dir, 'subids_%s.npy' % (result_file_name))
    linear_path = os.path.join(args.res_dir, 'linear_out_%s.npy' % (result_file_name))
    output_npy(hidden_path, test_out_hiddens, args=args)
    output_npy(subids_path, test_out_subids, args=args)
    output_npy(linear_path, test_out_linear, args=args)

    cal_regression_score(args, logger, args.res_dir, result_file_name, args.clinical_score_index,
                         best_true_clini_scores, best_true_clini_score_labels, best_pred_clini_scores)

    return loss.mean().item(), acc_max.item(), acc[:, argmax].std().item()


def cross_validation_without_val_set( dataset,
                                      args,
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
                                      result_file_name=None,
                                      json_path=None,
                                      disease_id=0,
                                      isCrossAtten=False,
                                      isSoftSimilarity=False,
                                      rbf_gamma=0.005,
                                      num_classes=2,
                                      lambda_loss=None,
                                      num_regr=4,
                                      model4eachregr = False,
                                      scaler4score=None
                                      ):
    if lambda_loss is None:
        lambda_loss = [1.0, 1.0, 1.0, 0.0000025, 0.2, 0.2]
    num_rois = 90
    feature_dim = 3
    if args.isMultiFusion:
        num_rois = num_rois*feature_dim
        feature_dim = 1
    go_snps, adj, pool_dim, n_l, go_level, new_go_ids, go_ids_genes_list = parse_go_json(json_path)
    A = gpu(torch.tensor(adj).float().t().to_sparse().coalesce(), device)
    A_g = gpu(torch.tensor(go_snps).float().to_sparse().coalesce(), device)
    pool_dim = np.asarray(pool_dim).tolist()
    l_dim = 32

    test_out_hiddens = []
    test_out_subids = []
    test_out_linear= []
    score_result = []
    test_losses, accs, durations = [], [], []
    count = 1
    best_true_clini_scores = []
    best_true_clini_score_labels = []
    best_pred_clini_scores = []

    for fold, (train_idx, test_idx, val_idx) in enumerate(
            zip(*k_fold(dataset, folds, args))):
        print("CV fold " + str(count))
        count += 1

        train_idx = torch.cat([train_idx, val_idx], 0)
        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]
        train_dataset, test_dataset = KNNImputation(args, train_dataset, test_dataset, scaler4score)

        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)#False, sampler=ImbalancedDatasetSampler(train_dataset)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

        model = SGCN_GCN_IMGSNP(num_layers, hidden, A_g, A, pool_dim, l_dim, device, rois=num_rois, H_0=feature_dim, num_classes=num_classes, isSoftSimilarity=isSoftSimilarity, rbf_gamma=rbf_gamma, isCrossAtten=isCrossAtten,num_regr=num_regr,
                                model4eachregr=model4eachregr, isuseFeat4Regr=args.isuseFeat4Regr, isImageOnly=args.isImageOnly, isSNPsOnly=args.isSNPsOnly, isMultiFusion=args.isMultiFusion)
        model = model.to(device)
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        criterion_recon = nn.MSELoss(reduction='none')
        temperature = gpu_ts(0.1, device)
        score_result_epoch = []
        t_start = time.perf_counter()

        best_loss = np.inf
        best_corr_regr = -np.inf
        best_all_out_hidden = []
        best_all_out_subid = []
        best_all_out_linear = []
        best_true_clini_score = []
        best_true_clini_score_label = []
        best_pred_clini_score = []
        pbar = tqdm(range(1, epochs + 1), ncols=70)
        for epoch in pbar:
            train_loss = train(model, optimizer, train_loader, temperature, lambda_loss, criterion_recon, isSoftSimilarity, device)
            test_losses.append(eval_loss(model, test_loader, temperature, lambda_loss, criterion_recon, isSoftSimilarity,device))
            accs.append(eval_acc(model, test_loader, temperature, lambda_loss, criterion_recon, isSoftSimilarity,device))
            true_label, pred_label, acuracy, auc, test_f1, sensitivity, specificity, all_out_hidden, all_out_subid, all_out_linear, regression_result = eval_scores(model, test_loader, temperature, lambda_loss, criterion_recon, isSoftSimilarity,device, num_classes=num_classes,num_regr=num_regr)
            true_clini_score, pred_clini_score, corr, r2, mse = regression_result
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
            print_log = 'Fold: %d, epoch:%d, train_loss: %0.4f, test_loss: %0.4f, test_acc: %0.4f, test_auc: %0.4f, test_f1: %0.4f, test_sen: %0.4f, test_spe: %0.4f,' \
                        ' ' % (
                fold, eval_info["epoch"], eval_info["train_loss"], eval_info["test_loss"], eval_info["test_acc"], eval_info["test_auc"], eval_info["test_f1"]
                , eval_info["test_sen"], eval_info["test_spe"]
            )
            if args.clinical_score_index == -1:
                scores_name = ['tau', 'adas13', 'mmse'] #'tau',
                for each_regr in range(len(corr)):
                    print_log += '; %s corr: %.5f, r2: %.5f, mse: %.5f' % (
                    scores_name[each_regr], corr[each_regr], r2[each_regr], mse[each_regr])
            else:
                scores_name = ['label', 'age', 'edu', 'sex', 'abeta', 'tau', 'ptau', 'adas13', 'mmse']
                print_log += '; %s corr: %.5f, r2: %.5f, mse: %.5f' % (
                    scores_name[args.clinical_score_index], corr[0], r2[0], mse[0])

            print(print_log)
            if logger is not None:
                logger(print_log)

            if epoch % lr_decay_step_size == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_decay_factor * param_group['lr']
            score_result_epoch.append([eval_info["test_acc"], eval_info["test_auc"], eval_info["test_f1"], eval_info["test_sen"], eval_info["test_spe"]])

            if best_loss>eval_info["test_loss"]:
                best_loss=eval_info["test_loss"]
                best_all_out_hidden = all_out_hidden
                best_all_out_subid = all_out_subid
                best_all_out_linear = all_out_linear
                best_true_clini_score = true_clini_score
                best_true_clini_score_label = true_label
                best_pred_clini_score = pred_clini_score
                model_path = os.path.join(args.res_dir, 'gcn_state_dict_%s_fold_%d.pt' % (result_file_name, fold))
                torch.save(model.state_dict(), model_path)

            if args.isPermutTest:
                best_true_clini_score = true_clini_score
                best_true_clini_score_label = true_label
                best_pred_clini_score = pred_clini_score

        if logger is not None:
            logger(log)
        score_result.append(score_result_epoch)

        output_importance(args, result_file_name, model, fold)

        test_out_hiddens.append(best_all_out_hidden)
        test_out_subids.append(best_all_out_subid)
        test_out_linear.append(best_all_out_linear)

        best_true_clini_scores.append(best_true_clini_score)
        best_true_clini_score_labels.append(best_true_clini_score_label)
        best_pred_clini_scores.append(best_pred_clini_score)

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
    output_npy(result_path, score_result, args=args)

    test_out_hiddens = torch.cat(test_out_hiddens).numpy()
    test_out_subids = torch.cat(test_out_subids).numpy()
    test_out_linear = torch.cat(test_out_linear).numpy()
    hidden_path = os.path.join(args.res_dir, 'hidden_%s.npy' % (result_file_name))
    subids_path = os.path.join(args.res_dir, 'subids_%s.npy' % (result_file_name))
    linear_path = os.path.join(args.res_dir, 'linear_out_%s.npy' % (result_file_name))
    output_npy(hidden_path, test_out_hiddens, args=args)
    output_npy(subids_path, test_out_subids, args=args)
    output_npy(linear_path, test_out_linear, args=args)

    cal_regression_score(args, logger, args.res_dir, result_file_name, args.clinical_score_index, best_true_clini_scores, best_true_clini_score_labels, best_pred_clini_scores)

    return loss.mean().item(), acc_max.item(), acc[:, argmax].std().item()

def cal_regression_score(args, logger, res_dir, result_file_name, clinical_score_index, best_true_clini_scores, best_true_clini_score_labels, best_pred_clini_scores):
    best_true_clini_scores = np.concatenate(best_true_clini_scores)
    best_pred_clini_scores = np.concatenate(best_pred_clini_scores)
    best_true_clini_score_labels = np.concatenate(best_true_clini_score_labels)
    for i in range(best_true_clini_scores.shape[1]):
        corr, pval = pearsonr(best_true_clini_scores[:,i], best_pred_clini_scores[:,i])
        mse = mean_squared_error(best_true_clini_scores[:,i], best_pred_clini_scores[:,i], squared=False)
        r2 = r2_score(best_true_clini_scores[:,i], best_pred_clini_scores[:,i])
        if args.clinical_score_index == -1:
            scores_name = ['tau','adas13', 'mmse'] #['label', 'age', 'edu', 'sex', 'abeta', 'tau', 'ptau', 'adas13', 'mmse']
            print_log = 'Regression for all clinical score %s: correlation: %.5f, r2: %.5f, mse: %.5f' % (
            scores_name[i], corr, r2, mse)
        else:
            scores_name = ['label', 'age', 'edu', 'sex', 'abeta', 'tau', 'ptau', 'adas13', 'mmse']
            print_log = 'Regression for all clinical score %s: correlation: %.5f, r2: %.5f, mse: %.5f' % (
            scores_name[args.clinical_score_index], corr, r2, mse)
        print(print_log)
        if logger is not None:
            logger(print_log)
        regression_true_path = os.path.join(res_dir, 'score_true_%s_%s.npy' % (scores_name[i], result_file_name))
        regression_true_label_path = os.path.join(res_dir, 'score_true_label_%s_%s.npy' % (scores_name[i], result_file_name))
        regression_pred_path = os.path.join(res_dir, 'score_pred_%s_%s.npy' % (scores_name[i], result_file_name))
        output_npy(regression_true_path, best_true_clini_scores, args=args)
        output_npy(regression_true_label_path, best_true_clini_score_labels, args=args)
        output_npy(regression_pred_path, best_pred_clini_scores, args=args)


def k_fold(dataset, folds, args):
    skf = StratifiedKFold(folds, shuffle=True, random_state=args.seed)

    test_indices, train_indices = [], []
    # tmp_label = [dataset[index].y.item() for index in range(len(dataset))]
    for _, idx in skf.split(torch.zeros(len(dataset)), dataset.data.y[dataset.indices()]):  #
        test_indices.append(torch.from_numpy(idx).long())

    val_indices = [test_indices[i - 1] for i in range(folds)]

    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.uint8)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero().view(-1))

    return train_indices, test_indices, val_indices


def k_fold2(dataset, folds, args):
    kf = KFold(folds, shuffle=True, random_state=args.seed)

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


def train(model, optimizer, loader, temperature, lambda_loss, criterion_recon, isSoftSimilarity, device, num_cluster=2):
    model.train()

    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        data = data.to(device)
        for param in model.parameters():
            param.requires_grad = True

        out, snps_hat, out_feat, out_lin,_, our_reg = model(data, temperature, device)
        loss_ce = lambda_loss[0] * F.nll_loss(out, data.y.view(-1))
        out_prob, snps_hat_prob, out_feat_prob, out_lin_prob,_, our_reg_prob = model(data, temperature, device, isExplain=True)
        loss_mi = lambda_loss[0] * F.nll_loss(out_prob, data.y.view(-1))

        loss_reg = lambda_loss[1] * (F.mse_loss(our_reg.view(-1), data.clini_score.view(-1))+F.mse_loss(our_reg_prob.view(-1), data.clini_score.view(-1)))/2

        loss_prob = lambda_loss[2] * model.loss_probability(data.x, data.edge_index, data.edge_attr, hp)

        recon_loss = lambda_loss[3] * (torch.sum(criterion_recon(snps_hat, data.snps_feat)) + torch.sum(criterion_recon(snps_hat_prob, data.snps_feat)))/2

        cluster_loss = 0
        if isSoftSimilarity:
            cluster_loss += lambda_loss[4] *(model.consist_loss(out_feat, data.tsne_fdim) + model.consist_loss(out_feat_prob, data.tsne_fdim))/2
        else:
            for c in range(num_cluster):
                cluster_loss += lambda_loss[4]*(model.consist_loss(out_feat[data.clust_y == c]) + model.consist_loss(out_feat_prob[data.clust_y == c]))/2
        orthogonal_loss = lambda_loss[5] * model.OrthogonalConstraint(out_feat)

        if lambda_loss[0]==0:
            loss_ce = 0.0
            loss_mi = 0.0
        loss = hp.lamda_ce * loss_ce + hp.lamda_mi * loss_mi + loss_reg + loss_prob + recon_loss + cluster_loss + orthogonal_loss
        # loss = F.nll_loss(out, data.y.view(-1))
        loss.backward()
        total_loss += loss.detach().cpu().item() * num_graphs(data)
        optimizer.step()
    return total_loss / len(loader.dataset)


def eval_acc(model, loader, temperature, lambda_loss, criterion_recon, isSoftSimilarity,device):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out, _, _, _,_,_ = model(data, temperature, device)
            pred = out.max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
    return correct / len(loader.dataset)


def eval_loss(model, loader, temperature, lambda_loss, criterion_recon, isSoftSimilarity,device,num_cluster=2):
    model.eval()

    loss = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out, snps_hat, out_feat, out_lin, linear_outf, our_reg = model(data, temperature, device)
            out_prob, snps_hat_prob, out_feat_prob, out_lin_prob, linear_outf_prob, our_reg_prob = model(data, temperature, device, isExplain=True)

        loss_ce = lambda_loss[0] * F.nll_loss(out, data.y.view(-1))  # , weight=weight
        loss_mi = lambda_loss[0] * F.nll_loss(out_prob, data.y.view(-1))
        loss_reg = lambda_loss[1] * (
                    F.mse_loss(our_reg.view(-1), data.clini_score.view(-1)) + F.mse_loss(our_reg_prob.view(-1),
                                                                                         data.clini_score.view(-1))) / 2
        recon_loss = lambda_loss[3] * (torch.sum(criterion_recon(snps_hat, data.snps_feat)) + torch.sum(
            criterion_recon(snps_hat_prob, data.snps_feat)))/2

        cluster_loss = 0
        if isSoftSimilarity:
            cluster_loss += lambda_loss[4]*(model.consist_loss(out_feat, data.tsne_fdim) + model.consist_loss(out_feat_prob, data.tsne_fdim))/2
        else:
            for c in range(num_cluster):
                cluster_loss += lambda_loss[4] * (model.consist_loss(out_feat[data.clust_y == c]) + model.consist_loss(
                    out_feat_prob[data.clust_y == c])) / 2
        orthogonal_loss = lambda_loss[5] * model.OrthogonalConstraint(out_feat)

        loss_prob = lambda_loss[2] * model.loss_probability(data.x, data.edge_index, data.edge_attr, hp)

        if lambda_loss[0]==0:
            loss_ce = 0.0
            loss_mi = 0.0
        # print(loss_ce, loss_mi, loss_reg, loss_prob, recon_loss, cluster_loss, orthogonal_loss)
        loss += (hp.lamda_ce * loss_ce + loss_reg + loss_prob + hp.lamda_mi * loss_mi + recon_loss + cluster_loss + orthogonal_loss).cpu().item() * num_graphs(data)


    return loss / len(loader.dataset)

def eval_scores(model, loader, temperature, lambda_loss, criterion_recon, isSoftSimilarity,device, num_classes=2, num_regr=4):
    model.eval()
    true_label = []
    pred_label = []
    true_clini_score = []
    pred_clini_score = []
    all_scores = []
    all_out_hidden = []
    all_out_linear = []
    all_out_subid = []
    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out, _, _, out_lin, linear_outf, our_reg = model(data, temperature, device)
            all_scores.append(out[:, 1].cpu().detach())
            pred = out.max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
        true_label = true_label + [per_label for per_label in data.y.cpu().numpy().tolist()]
        pred_label = pred_label + [per_label for per_label in pred.cpu().numpy().tolist()]
        batch_clini_score = torch.reshape(data.clini_score, (-1, num_regr))
        our_reg = torch.reshape(our_reg, (-1, num_regr))
        true_clini_score.append(batch_clini_score.cpu().detach())
        pred_clini_score.append(our_reg.cpu().detach())
        all_out_hidden.append(out_lin)
        all_out_linear.append(linear_outf)
        all_out_subid.append(data.sbjID.cpu())
    all_out_hidden = torch.cat(all_out_hidden).cpu()
    all_out_linear = torch.cat(all_out_linear).cpu()
    all_out_subid = torch.cat(all_out_subid).cpu()
    all_scores = torch.cat(all_scores).cpu().numpy()
    true_clini_score = torch.cat(true_clini_score)
    pred_clini_score = torch.cat(pred_clini_score)

    auc = 0
    try:
        if num_classes < 3:
            fpr, tpr, _ = metrics.roc_curve(np.asarray(true_label), all_scores, pos_label=1)
            auc = metrics.auc(fpr, tpr)
    except:
        auc = 0
    true_clini_score = true_clini_score.numpy()
    pred_clini_score = pred_clini_score.numpy()
    true_clini_score = np.asarray(true_clini_score)
    pred_clini_score = np.asarray(pred_clini_score)

    pred_clini_score[np.isnan(pred_clini_score)]=0
    corr, pval, mse, r2 = [],[],[],[]
    for i in range(true_clini_score.shape[1]):
        corr_each, pval_each = pearsonr(true_clini_score[:,i], pred_clini_score[:,i])
        mse_each = mean_squared_error(true_clini_score[:,i], pred_clini_score[:,i], squared=False)
        r2_each = r2_score(true_clini_score[:,i], pred_clini_score[:,i])
        corr.append(corr_each)
        pval.append(pval_each)
        mse.append(mse_each)
        r2.append(r2_each)
    regression_result = (true_clini_score.tolist(), pred_clini_score.tolist(), corr, r2, mse)

    true_label = np.asarray(true_label)
    pred_label = np.asarray(pred_label)
    test_f1 = f1_score(true_label, pred_label, average='weighted')
    if num_classes<3:
        cm = confusion_matrix(true_label, pred_label)
        TN, FP, FN, TP = cm.ravel()
        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)
    else:
        sensitivity=0
        specificity=0
    return true_label, pred_label, correct / len(loader.dataset), auc, test_f1, sensitivity, specificity, all_out_hidden, all_out_subid, all_out_linear, regression_result