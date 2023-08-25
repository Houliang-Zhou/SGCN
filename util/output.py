import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv, global_add_pool, global_mean_pool, global_sort_pool, global_max_pool
from torch_geometric.utils import to_dense_batch
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn import init
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool, global_add_pool
from torch.nn import Linear
import os

def output_npy(path, data, args=None):
    if args is not None and args.isPermutTest:
        return
    if path is not None:
        with open(path, 'wb') as f:
            np.save(f, data)


def output_importance(args, result_file_name, model, fold):
    node_importance = model.prob.detach().cpu().numpy()
    snps_importance = model.snps_prob.detach().cpu().numpy()
    prob_bias = model.prob_bias.detach().cpu().numpy()
    node_importance_path = os.path.join(args.res_dir, 'node_importance_%s_fold_%d.npy' % (result_file_name, fold))
    prob_bias_path = os.path.join(args.res_dir, 'edge_prob_bias_%s_fold_%d.npy' % (result_file_name, fold))
    snps_importance_path = os.path.join(args.res_dir, 'snps_importance_%s_fold_%d.npy' % (result_file_name, fold))
    output_npy(node_importance_path, node_importance, args=args)
    output_npy(snps_importance_path, snps_importance, args=args)
    output_npy(prob_bias_path, prob_bias, args=args)
    print("save node importance for fold %d: "%(fold), node_importance.shape)
    print("save snps importance for fold %d: "%(fold), snps_importance.shape)
    print("save prob bias for fold %d: " % (fold), prob_bias.shape)

