#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 13:16:52 2021

@author: sayan
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
from util.convert_to_gpu import gpu
from util.convert_to_gpu_and_tensor import gpu_t
from util.convert_to_gpu_scalar import gpu_ts
from util.convert_to_cpu import cpu
import collections
from torch_scatter import scatter, scatter_add


class Gene_ontology_network(nn.Module):
    def __init__(self, A_g, A, in_f_dim, n_l, f_dim, pool_dim, l_dim, device, dim_snps_atten=5):
        """

        :param A_g: gene x GO: 20 x 10
        :param A: GO x GO: 10 x 10
        :param in_f_dim: input feature dimension 2
        :param n_l: number of layers 2
        :param f_dim: dim of hidden layers [5,5]
        :param pool_dim: [[3,6,11]]
        :param l_dim: 5
        """
        super(Gene_ontology_network, self).__init__()
        self.device=device

        # converting the nework to sparse matrix

        # storing nnz locations for future use

        self.n_loc_in = []
        self.store_in = []
        ind_pool = [0] + pool_dim[0]  # [0, pool_dim]

        A = A
        A_g = A_g

        A_t = A.t().coalesce()

        for i in range(n_l):
            T = A.to_dense()[ind_pool[i]:,
                :]  # This operation discards the leaf nodes and again identify the NNZ locations.
            T = T[:, ind_pool[i]:]
            # print(T.shape)
            T = T.to_sparse()
            A = gpu(T.clone(), device)
            temp = T.coalesce().indices()
            temp1 = temp[0, :]
            self.n_loc_in.append(T.coalesce().indices())  # change   .coalesce()
            self.store_in.append(self.store_ind(self.n_loc_in[i][0, :], device))

            # This set of codes are for the decoding operation.

        self.n_loc_out = []
        self.store_out = []
        ind_pool = pool_dim[0]
        for i in range(n_l):
            temp = sum(ind_pool[:n_l - i - 1])
            T = A_t.to_dense()[sum(ind_pool[:n_l - i - 1]):, :]
            T = T[:, sum(ind_pool[:n_l - i]):]
            T = T.to_sparse().coalesce()
            self.n_loc_out.append(T.indices())
            self.store_out.append(self.store_ind(self.n_loc_out[i][0, :], device))

        #############################################################################################################################################3
        # gene encoding
        self.i = A_g.indices().detach()
        self.size = A_g.size()
        self.t = nn.ParameterList(
            [nn.Parameter(torch.Tensor(A_g.values().size()[0], ).data.normal_(1.0, 0.1)) for ii in range(in_f_dim)])

        # gene decoding
        A_g_t = A_g.t().coalesce()
        self.i_D = A_g_t.indices().detach()
        self.size_D = A_g_t.size()

        self.t_D = nn.ParameterList([nn.Parameter(torch.Tensor(A_g_t.values().size()[0], ).data.normal_(1.0, 0.1))])

        ##############################################################################################################################################
        f_dim = [in_f_dim] + f_dim

        self.pool = pool_dim[0]

        self.f_dim = f_dim

        # forward gcn
        self.w_inc = nn.ModuleList([nn.Linear(f_dim[i], f_dim[i + 1], bias=False) for i in range(n_l)])
        self.w_s_loop = nn.ModuleList([nn.Linear(f_dim[i], f_dim[i + 1], bias=False) for i in range(n_l)])
        self.w_att_s = nn.ModuleList([nn.Linear(f_dim[i + 1], 1, bias=False) for i in range(n_l)])
        self.w_att_s_act = nn.ModuleList([nn.Sigmoid() for i in range(n_l)])
        self.G_B = nn.ModuleList([nn.LayerNorm(sum(self.pool[i:])) for i in range(n_l)])
        self.w_act = nn.ModuleList([nn.ReLU() for i in range(n_l)])
        self.gcn_D = nn.ModuleList([nn.Dropout2d(0.4) for i in range(n_l)])
        self.w_att_in = nn.ModuleList([nn.Linear(2 * f_dim[i + 1], 1, bias=False) for i in range(n_l)])
        self.w_att_in_act = nn.ModuleList([nn.Tanh() for i in range(n_l)])

        # reverse decoding gcn
        self.w_out = nn.ModuleList([nn.Linear(f_dim[i], f_dim[i - 1], bias=False) for i in range(n_l, 0, -1)])
        self.w_s_loop_out = nn.ModuleList([nn.Linear(f_dim[i], f_dim[i - 1], bias=False) for i in range(n_l, 0, -1)])
        self.G_B_D = nn.ModuleList([nn.LayerNorm(sum(self.pool[i:])) for i in range(n_l - 1, -1, -1)])
        self.w_act_out = nn.ModuleList([nn.ReLU() for i in range(n_l)])
        self.gcn_D_D = nn.ModuleList([nn.Dropout2d(0.4) for i in range(n_l)])

        ##############################################################################################################################################

        self.conc_for_attention = nn.Sequential(
            nn.Linear(f_dim[-1], dim_snps_atten, bias=False),
            nn.BatchNorm1d(sum(self.pool) - sum(self.pool[0:n_l])),
            nn.ReLU()
        )

        self.conc = nn.Linear(f_dim[-1], 1, bias=False)  # This is the read out phase
        self.B = nn.Sequential(
            nn.BatchNorm1d(sum(self.pool) - sum(self.pool[0:n_l])),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.conc_D = nn.Linear(f_dim[0], 1, bias=False)

        self.B_D = nn.Sequential(
            nn.BatchNorm1d(sum(self.pool)),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        # This one is \mathcal{E}_g
        self.latent = nn.Sequential(
            nn.Linear((sum(self.pool) - sum(self.pool[0:n_l]))*1, 32, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, l_dim, bias=False),
            nn.BatchNorm1d(l_dim),
            nn.ReLU(),
        )
        # The classification module.
        self.classification = nn.Sequential(
            nn.BatchNorm1d(l_dim + 54),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(l_dim + 54, 16, bias=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 1, bias=True),
            nn.Sigmoid()
        )

    ##############################################################################################################################

    def store_ind(self, indices, device):
        st = collections.Counter(indices.to('cpu').tolist())
        store = []
        vv = list(st.values())
        for k in range(len(st)):
            store = store + [k] * vv[k]

        return gpu(torch.tensor(store), device)

    def gcn(self, A, t):
        return torch.sparse.mm(A, t)

    def attention_adj(self, ii, vv, size, store, i, device):
        # This function generates the importance matrix, E

        T = torch.sparse.sum(torch.sparse.FloatTensor(ii, vv, size), dim=1)
        div = T.values()[store]
        V = vv / div
        A = torch.sparse.FloatTensor(i, V, size)
        return A, i, V, size

    def helper(self, x1, x2, W_att, W_att_act):

        #        t1_out =  torch.cat((cpu(x1)  , cpu(x2)), dim=2)
        t1_out = torch.cat((x1, x2), dim=2)
        return torch.exp(W_att_act(W_att(t1_out)))

    def create_batch_A(self, i_in, n, M, store_in, K):

        out_in = i_in
        batch_store_in = store_in
        for i in range(1, n):
            out_in = torch.cat((out_in, i_in.clone() + i * M), dim=1)
            batch_store_in = torch.cat((batch_store_in, store_in.clone() + i * K), dim=0)
        return out_in, batch_store_in

    def batch_mul(self, A, x, i, size):
        x_temp = x.clone()
        o = A.unsqueeze(0).unsqueeze(2) * x_temp[:, i[1, :], :]
        out = scatter(o.clone(), i[0, :], dim=1, reduce="sum", out=gpu(torch.zeros(size), self.device))
        return out

    ##########################################################################################################################################

    def forward(self, data, T, device):

        # gene encoding
        W = torch.sparse.FloatTensor(self.i, self.t[0], self.size)
        x = torch.sparse.mm(W, data.t()).t()
        x = x.unsqueeze(2)

        for ii in range(1, len(self.t)):
            W = torch.sparse.FloatTensor(self.i, self.t[ii], self.size)
            t = torch.sparse.mm(W, data.t()).t()
            x = torch.cat((x, t.unsqueeze(2)), dim=2)

            #########################################################################################33
        # now x has dim n_subjects X no of nodes X channels
        for jj in range(len(self.w_inc)):

            out = gpu(torch.zeros((x.size()[0], x.size()[1], self.f_dim[jj + 1])), device)

            i_in = self.n_loc_in[jj].clone()
            store_in = self.store_in[jj]

            x_in = self.w_inc[jj](x)  # Multiplication of feature with GCN filter.
            x_s = self.w_s_loop[jj](x)  # Self influence

            x1_temp = x_in.clone()
            x2_temp = x_in.clone()

            v_inc = self.helper(x1_temp[:, i_in[0, :].clone(), :], x2_temp[:, i_in[1, :].clone(), :], self.w_att_in[jj],
                                self.w_att_in_act[jj])  # Generating the edge importance between each pair of nodes
            v_s = self.w_att_s_act[jj](self.w_att_s[jj](x_s))  # Self importance, $\beta$

            for k in range(x.size()[0]):
                A_hat_in, _, _, _ = self.attention_adj(i_in, v_inc[k, :].squeeze(),
                                                       torch.Size([x.size()[1], x.size()[1]]), store_in, i_in, device=device)

                x_incoming = self.gcn(A_hat_in, x_in[k, :, :])  # n_sub x no of nodes x   d
                x_self = x_s[k, :, :] * v_s[k, :, :]

                t = x_incoming + x_self
                out[k, :, :] = t

            temp = self.G_B[jj](out.permute(0, 2, 1)).permute(0, 2, 1)
            out1 = self.gcn_D[jj](self.w_act[jj](temp))

            ind_pool = self.pool[jj]

            x = out1[:, ind_pool:, :].clone()  # We are removing the leaf nodes due to hierarchical pooling.
            # print(x.shape)

        atten_out = self.conc_for_attention(x)
        inp_out = self.B(self.conc(x).squeeze())

        # Decoding Operation
        for jj in range(len(self.w_inc)):
            i_out = self.n_loc_out[jj].clone()
            store_out = self.store_out[jj]

            x_out = self.w_out[jj](x)
            x_s_out = self.w_s_loop_out[jj](x)
            _, _, v_out, _ = self.attention_adj(i_out, gpu(torch.ones(i_out.size()[1]), device),
                                                torch.Size([sum(self.pool[len(self.w_inc) - jj - 1:]), x.size()[1]]),
                                                store_out, i_out, device=device)

            x_outgoing = self.batch_mul(v_out, x_out, i_out, torch.Size(
                [x_out.size()[0], sum(self.pool[len(self.w_inc) - jj - 1:]), x_out.size()[2]]))
            x_self_out = gpu(torch.zeros(x_outgoing.size()), device)
            x_self_out[:, self.pool[len(self.w_inc) - jj - 1]:, :] = x_s_out
            out_decoder = x_outgoing + x_self_out
            out_temp = self.gcn_D_D[jj](
                self.w_act_out[jj](self.G_B_D[jj](out_decoder.permute(0, 2, 1)).permute(0, 2, 1)))
            x = out_temp.clone()

        # print(out_temp.shape)
        out_D = self.B_D(self.conc_D(out_temp).squeeze())

        # gene decoding
        W_D = torch.sparse.FloatTensor(self.i_D, self.t_D[0], self.size_D)
        x_D = torch.sparse.mm(W_D, out_D.t()).t()

        # Latent Projection.
        latent = self.latent(inp_out.view((inp_out.shape[0],-1)))

        return latent, x_D, [gpu(torch.zeros(3), device)], atten_out


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A = np.random.randint(2, size=(20, 20))
    A = gpu(torch.tensor(A).float().t().to_sparse().coalesce(), device)
    A_g = np.random.randint(2, size=(20, 54))
    A_g = gpu(torch.tensor(A_g).float().to_sparse().coalesce(), device)
    x = gpu_t(np.random.randn(30, 54), device)
    pool_dim = np.asarray([[3, 6, 11]]).tolist()
    l_dim = 5
    t = gpu_ts(0.1, device)
    net = Gene_ontology_network(A_g, A, 2, 2, [5, 5], pool_dim, l_dim, device=device)
    net = net.to(device)
    latent, x_hat, prob = net(x, t, device)
    print(latent.shape, x_hat.shape)