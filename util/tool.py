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
from sklearn.impute import KNNImputer

def KNNImputation(args, train_dataset, test_dataset, scaler4score, k=3):
    demographics_train = []
    clini_score_train = []
    demographics_test = []
    clini_score_test = []
    for eachdata in train_dataset:
        demographics_train.append(eachdata.demographics.unsqueeze(0))
        clini_score_train.append(eachdata.clini_score.unsqueeze(0))
    for eachdata in test_dataset:
        demographics_test.append(eachdata.demographics.unsqueeze(0))
        clini_score_test.append(eachdata.clini_score.unsqueeze(0))
    demographics_train = torch.cat(demographics_train).numpy()
    clini_score_train = torch.cat(clini_score_train).numpy()
    demographics_test = torch.cat(demographics_test).numpy()
    clini_score_test = torch.cat(clini_score_test).numpy()
    imputer = KNNImputer(n_neighbors=k)
    demographics_train = imputer.fit_transform(demographics_train)
    demographics_test = imputer.transform(demographics_test)
    demographics_train = scaler4score.transform(demographics_train)
    demographics_test = scaler4score.transform(demographics_test)
    if args.clinical_score_index == -1:
        select_index = np.array([5, 7, 8])
        for i in range(len(train_dataset)):
            clini_score = torch.Tensor(demographics_train[i, select_index]).float()
            train_dataset[i].clini_score = clini_score
        for i in range(len(test_dataset)):
            clini_score = torch.Tensor(demographics_test[i, select_index]).float()
            test_dataset[i].clini_score = clini_score
    else:
        select_index = np.array([args.clinical_score_index])
        for i in range(len(train_dataset)):
            clini_score = torch.Tensor(demographics_train[i, select_index]).float()
            train_dataset[i].clini_score = clini_score
        for i in range(len(test_dataset)):
            clini_score = torch.Tensor(demographics_test[i, select_index]).float()
            test_dataset[i].clini_score = clini_score
    return train_dataset, test_dataset