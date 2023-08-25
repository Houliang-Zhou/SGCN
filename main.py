import os.path as osp
import os, sys
import time
from shutil import copy, rmtree
from itertools import product
import pdb
import argparse
import random
import torch
import numpy as np
from kernel.datasets import get_dataset
from kernel.train_eval_sgcn import cross_validation_with_val_set
from kernel.train_eval_sgcn import cross_validation_without_val_set
from kernel.gcn import *
from kernel.graph_sage import *
from kernel.gin import *
from kernel.gat import *
from kernel.sgcn import *
from kernel.graclus import Graclus
from kernel.top_k import TopK
from kernel.diff_pool import *
from kernel.global_attention import GlobalAttentionNet
from kernel.set2set import Set2SetNet
from kernel.sort_pool import SortPool
from sgcn_data import loadBrainImg, ADNIDataset
import sgcn_hyperparameters as hp
from utils import create_subgraphs, return_prob
from util_gdc import preprocess_diffusion

# used to traceback which code cause warnings, can delete
import traceback
import warnings
import sys
def warn_with_traceback(message, category, filename, lineno, file=None, line=None):

    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))

warnings.showwarning = warn_with_traceback


# General settings.
parser = argparse.ArgumentParser(description='SGCN for ADNI graphs')
parser.add_argument('--data', type=str, default='SGCN')
parser.add_argument('--clean', action='store_true', default=False,
                    help='use a cleaned version of dataset by removing isomorphism')
parser.add_argument('--no_val', action='store_true', default=False,
                    help='if True, do not use validation set, but directly report best\
                    test performance.')
parser.add_argument('--knn', type=int, default=5,
                    help='k for knn graph')
parser.add_argument('--disease_id', type=int, default=0,
                    help='disease_id for classification: 0, 1, 2')

parser.add_argument('--isTestAdnitype', action='store_true', default=False,
                    help='is TestAdnitype')
parser.add_argument('--isShowValResult', action='store_true', default=True,
                    help='is ShowValResult')
parser.add_argument('--adnitype_id', type=int, default=0)
parser.add_argument('--disease_id4Adnitype', type=int, default=1)

# GNN settings.
parser.add_argument('--model', type=str, default='SGCN_GCN',
                    help='SGCN, GCN, GraphSAGE, GIN, GAT')
parser.add_argument('--layers', type=int, default=3)
parser.add_argument('--hiddens', type=int, default=10)

# Training settings.
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=1E-3)
parser.add_argument('--lr_decay_factor', type=float, default=0.5)
parser.add_argument('--lr_decay_step_size', type=int, default=50)
parser.add_argument('--fold', type=int, default=5)

# Other settings.
parser.add_argument('--seed', type=int, default=1000)
parser.add_argument('--search', action='store_true', default=False,
                    help='search hyperparameters (layers, hiddens)')
parser.add_argument('--save_appendix', default='',
                    help='what to append to save-names when saving results')
parser.add_argument('--cpu', action='store_true', default=False, help='use cpu')
parser.add_argument('--cuda', type=int, default=0, help='which cuda to use')
args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

file_dir = os.path.dirname(os.path.realpath('__file__'))
if args.save_appendix == '':
    args.save_appendix = '_' + time.strftime("%Y%m%d%H%M%S")
args.res_dir = os.path.join(file_dir, 'results/ADNI{}'.format(args.save_appendix))
print('Results will be saved in ' + args.res_dir)
if not os.path.exists(args.res_dir):
    os.makedirs(args.res_dir)

# save command line input
cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
with open(os.path.join(args.res_dir, 'cmd_input.txt'), 'a') as f:
    f.write(cmd_input)
print('Command line input: ' + cmd_input + ' is saved.')

if args.data == 'all':
    datasets = [ 'DD', 'MUTAG', 'PROTEINS', 'PTC_MR', 'ENZYMES']
else:
    datasets = [args.data]

if args.search:
    if args.h is None:
        layers = [3, 3, 3, 3]
        hiddens = [32]
    else:
        layers = [3, 3, 3, 3]
        hiddens = [32, 16, 10, 5]
else:
    layers = [args.layers]
    hiddens = [args.hiddens]

if args.model == 'all':
    #nets = [GCN, GraphSAGE, GIN, GAT]
    nets = [NestedGCN, NestedGraphSAGE, NestedGIN, NestedGAT]
else:
    nets = [eval(args.model)]

def logger(info):
    f = open(os.path.join(args.res_dir, 'log.txt'), 'a')
    print(info, file=f)

device = torch.device(
    'cuda:%d'%(args.cuda)  if torch.cuda.is_available() and not args.cpu else 'cpu'
)
print(device)

if args.no_val:
    cross_val_method = cross_validation_without_val_set
else:
    cross_val_method = cross_validation_with_val_set

results = []
for dataset_name, Net in product(datasets, nets):
    best_result = (float('inf'), 0, 0)
    log = '-----\n{} - {}'.format(dataset_name, Net.__name__)
    print(log)
    logger(log)
    combinations = product(layers, hiddens)
    for num_layers, hidden in combinations:
        log = "Using {} layers, {} hidden units".format(num_layers, hidden)
        print(log)
        logger(log)
        result_file_name = "result_sgcn_layers{}_hidden{}".format(num_layers, hidden)
        result_path = os.path.join(args.res_dir, '%s.npy'%(result_file_name))
        max_nodes_per_hop = None
        data_path = './data/brain_image/knn/%d/'%(args.knn)
        adni_dataset = loadBrainImg(disease_id=args.disease_id, isShareAdj=hp.isShareAdj, isInfo_Score=hp.isInfo_Score,
                               isSeperatedGender=hp.isSeperatedGender, selected_gender=hp.selected_gender,
                                    data_path=data_path)
        if args.isTestAdnitype:
            dataset = adni_dataset
        else:
            dataset = ADNIDataset('./', 'SGCN', adni_dataset)
        model = Net(hp.H_0, hidden, hidden, hp.H_3)
        loss, acc, std = cross_val_method(
            args,
            dataset,
            model,
            folds=args.fold,
            epochs=args.epochs,
            batch_size=hp.batch_size,
            lr=hp.learning_rate,
            lr_decay_factor=args.lr_decay_factor,
            lr_decay_step_size=args.lr_decay_step_size,
            weight_decay=0,
            device=device,
            logger=logger,
            result_path=result_path,
            pre_transform=None)
        if loss < best_result[0]:
            best_result = (loss, acc, std)
            best_hyper = (num_layers, hidden)

    desc = '{:.3f} ± {:.3f}'.format(
        best_result[1], best_result[2]
    )
    log = 'Best result - {}, with {} layers and {} hidden units and h = {}'.format(
        desc, best_hyper[0], best_hyper[1], best_hyper[2]
    )
    print(log)
    logger(log)
    results += ['{} - {}: {}'.format(dataset_name, model.__class__.__name__, desc)]

log = '-----\n{}'.format('\n'.join(results))
print(cmd_input[:-1])
print(log)
logger(log)
