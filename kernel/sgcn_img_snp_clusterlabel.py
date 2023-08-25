import math
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv, global_add_pool, global_mean_pool, global_sort_pool, global_max_pool
from torch_geometric.utils import to_dense_batch
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn import init
from torch_geometric.nn import GATConv, global_mean_pool
from torch.nn import Linear
from kernel.go_model import *


class SGCN_GCN_CLUSTERLABEL(torch.nn.Module):

    def __init__(self, num_layers, hidden, A_g, A, pool_dim, l_dim, device, *args, hidden_linear=64, rois=90, H_0=1, num_features=1, num_classes=3, num_cluster=2, isCrossAtten=False, isPredictCluster = True, **kwargs):
        super(SGCN_GCN_CLUSTERLABEL, self).__init__()
        self.device=device
        self.isCrossAtten=isCrossAtten
        self.input = None
        self.final_conv_acts = None
        self.final_conv_grads = None
        self.isPredictCluster = isPredictCluster
        self.rois = rois
        self.prob_dim = H_0
        self.conv1 = GCNConv(num_features, hidden)
        self.convs = torch.nn.ModuleList()
        n_l=2
        dim_snps_atten = hidden
        if isCrossAtten:
            for i in range(num_layers - 2):
                self.convs.append(GCNConv(hidden, hidden))
                dim_snps_atten+=hidden
            self.convs.append(GCNConv(hidden, hidden))
            dim_snps_atten += hidden
            self.pool = pool_dim[0]
            linear_input = (sum(self.pool) - sum(self.pool[0:n_l]))*dim_snps_atten
            self.multihead_attn = torch.nn.MultiheadAttention(dim_snps_atten, 2, batch_first=True)
        else:
            for i in range(num_layers - 1):
                self.convs.append(GCNConv(hidden, hidden))
            linear_input = 90 * num_layers * hidden + l_dim

        # self.lin1 = torch.nn.Linear(90 * num_layers * hidden + linear_input + l_dim + 54, hidden_linear)
        # self.lin1 = torch.nn.Linear(2 * 90 * num_layers * hidden + l_dim + 54, hidden_linear)

        self.lin1_classify = torch.nn.Linear(90 * num_layers * hidden + l_dim, hidden_linear)
        # self.lin1_classify = torch.nn.Linear(90 * num_layers * hidden, hidden_linear)
        self.lin2_classify = Linear(hidden_linear, num_classes)

        self.lin1_cluster = torch.nn.Linear(90 * num_layers * hidden + l_dim, hidden_linear)
        # self.lin1_cluster = torch.nn.Linear(90 * num_layers * hidden, hidden_linear)
        self.lin2_cluster = Linear(hidden_linear, num_cluster)

        self.prob = Parameter(torch.empty((self.rois, self.prob_dim)))  # *0.5
        # init.kaiming_uniform_(self.prob, a=math.sqrt(5))
        self.prob_bias = Parameter(torch.empty((self.prob_dim * 2, 1)))
        init.kaiming_uniform_(self.prob_bias, a=math.sqrt(5))
        self.edge_prob = Parameter(torch.empty((self.rois, self.rois)))
        init.kaiming_uniform_(self.prob, a=math.sqrt(5))
        init.kaiming_uniform_(self.edge_prob, a=math.sqrt(5))

        self.snps_prob = Parameter(torch.empty((1, 54)))
        init.kaiming_uniform_(self.snps_prob, a=math.sqrt(5))

        self.go_network = Gene_ontology_network(A_g, A, 2, n_l, [5, 5], pool_dim, l_dim, device, dim_snps_atten=dim_snps_atten)

        self.batch_norm = torch.nn.BatchNorm1d(num_layers * hidden)


    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1_classify.reset_parameters()
        self.lin2_classify.reset_parameters()
        self.lin1_cluster.reset_parameters()
        self.lin2_cluster.reset_parameters()

        self.prob = Parameter(torch.empty((self.rois, self.prob_dim)))  # *0.5
        # init.kaiming_uniform_(self.prob, a=math.sqrt(5))
        self.prob_bias = Parameter(torch.empty((self.prob_dim * 2, 1)))
        init.kaiming_uniform_(self.prob_bias, a=math.sqrt(5))
        self.edge_prob = Parameter(torch.empty((self.rois, self.rois)))
        init.kaiming_uniform_(self.prob, a=math.sqrt(5))
        init.kaiming_uniform_(self.edge_prob, a=math.sqrt(5))

        self.snps_prob = Parameter(torch.empty((1, 54)))
        init.kaiming_uniform_(self.snps_prob, a=math.sqrt(5))


    def activations_hook(self, grad):
        self.final_conv_grads = grad

    def cal_probability(self, x, edge_index, edge_weight, snps_feat=None):
        N, D = x.shape
        x = x.reshape(N // self.rois, self.rois, D)
        x_prob = self.prob  # torch.sigmoid(self.prob) #self.prob
        x_feat_prob = x * x_prob
        x_feat_prob = x_feat_prob.reshape(N, D)
        # print(x_prob)

        conat_prob = torch.cat((x_feat_prob[edge_index[0]], x_feat_prob[edge_index[1]]), -1)
        edge_prob = torch.sigmoid(conat_prob.matmul(self.prob_bias)).view(-1)
        # edge_prob = (edge_prob + edge_prob.t()) / 2
        edge_weight_prob = edge_weight * edge_prob

        if snps_feat is not None:
            snps_prob = torch.sigmoid(self.snps_prob)
            snps_feat_prob = snps_feat*snps_prob
            return x_feat_prob, edge_weight_prob, x_prob, edge_prob, snps_feat_prob, snps_prob
        else:
            return x_feat_prob, edge_weight_prob, x_prob, edge_prob

    def loss_probability(self, x, edge_index, edge_weight, hp, eps=1e-6):
        _, _, x_prob, edge_prob = self.cal_probability(x, edge_index, edge_weight)

        x_prob = torch.sigmoid(x_prob)

        N, D = x_prob.shape
        all_num = (N * D)
        # f_sum_loss = torch.sum(x_prob)/all_num
        f_sum_loss = x_prob.norm(dim=-1, p=1).sum() / N
        f_entrp_loss = -torch.sum(
            x_prob * torch.log(x_prob + eps) + (1 - x_prob) * torch.log((1 - x_prob) + eps)) / all_num

        N = edge_prob.shape[0]
        all_num = N
        # e_sum_loss = torch.sum(edge_prob)/all_num
        e_sum_loss = edge_prob.norm(dim=-1, p=1) / N
        e_entrp_loss = -torch.sum(
            edge_prob * torch.log(edge_prob + eps) + (1 - edge_prob) * torch.log((1 - edge_prob) + eps)) / all_num

        snps_prob = torch.sigmoid(self.snps_prob)
        N, D = snps_prob.shape
        all_num = (N * D)
        # f_sum_loss = torch.sum(x_prob)/all_num
        snps_sum_loss = snps_prob.norm(dim=-1, p=1).sum() / N
        snps_entrp_loss = -torch.sum(
            snps_prob * torch.log(snps_prob + eps) + (1 - snps_prob) * torch.log((1 - snps_prob) + eps)) / all_num

        # sum_loss = (f_sum_loss+e_sum_loss+f_entrp_loss+e_entrp_loss)/4
        loss_prob = hp.lamda_x_l1 * f_sum_loss + hp.lamda_e_l1 * e_sum_loss + hp.lamda_x_ent * f_entrp_loss + hp.lamda_e_ent * e_entrp_loss + hp.lamda_x_l1 * snps_sum_loss + hp.lamda_x_ent * snps_entrp_loss

        return loss_prob

    def consist_loss(self, s):
        if len(s) == 0:
            return 0
        # s = torch.sigmoid(s)
        W = torch.ones(s.shape[0], s.shape[0])
        D = torch.eye(s.shape[0]) * torch.sum(W, dim=1)
        L = D - W
        L = L.to(self.device)
        cluster_loss = torch.trace(torch.transpose(s, 0, 1) @ L @ s) / (s.shape[0] * s.shape[0])
        return cluster_loss

    def forward(self, data, temperature, device, isExplain=False):
        x, edge_index, batch, edge_weight = data.x, data.edge_index, data.batch, data.edge_attr
        snps_feat = data.snps_feat
        x.requires_grad = True
        self.input = x

        if isExplain:
            x_prob, edge_weight_prob, _, _, snps_feat_prob, _ = self.cal_probability(x, edge_index, edge_weight, snps_feat)
        else:
            x_prob, edge_weight_prob, snps_feat_prob = x, edge_weight, snps_feat

        x = F.relu(self.conv1(x_prob, edge_index, edge_weight_prob))
        xs = [x]
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_weight_prob))
            xs += [x]
        # x = global_mean_pool(torch.cat(xs, dim=1), batch)
        # x = F.relu(self.lin1(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        # x = self.lin2(x)

        x = torch.cat(xs, dim=1)

        fill_value = x.min().item() - 1
        batch_x, _ = to_dense_batch(x, batch, fill_value)
        B, N, D = batch_x.size()
        img_out = batch_x.view(B, -1)
        # x = img_out

        latent, x_hat, _, atten_out = self.go_network(snps_feat_prob, temperature, device)
        # x_hat = snps_feat

        if self.isCrossAtten:
            attn_output, attn_output_weights = self.multihead_attn(batch_x, atten_out, atten_out)
            attn_output = F.relu(attn_output)
            x = attn_output #.reshape((attn_output.shape[0], -1))
        else:
            x = torch.cat((img_out, latent), -1)

        # if self.isCrossAtten:
        #     attn_output, attn_output_weights = self.multihead_attn(atten_out, batch_x, batch_x)
        #     x = attn_output.reshape((attn_output.shape[0], -1))
        # else:
        #     x = torch.cat((img_out, latent), -1)

        # x = torch.cat((img_out, x, F.relu(snps_feat), latent), -1)
        x = x.reshape((B, -1))
        x_prob = x_prob.reshape((B, -1))
        # print(torch.max(img_out),torch.max(x),torch.max(latent),torch.max(snps_feat_prob))

        # out_z = torch.cat((img_out, x, latent, snps_feat_prob), -1)
        out_z = torch.cat(((img_out+x)/2, latent), -1)
        # out_z = img_out

        # x = torch.permute(x, (0,2,1))
        # batch_x = torch.permute(batch_x, (0,2,1))
        # x = self.batch_norm(x+batch_x)
        # x = x.view(B, -1)
        # x = torch.cat((latent, x), -1)

        if self.isPredictCluster:
            x_cluster = F.relu(self.lin1_cluster(out_z))
        else:
            x_cluster = F.relu(self.lin1_cluster(torch.zeros_like(out_z).to(self.device)))
        x_cluster = F.dropout(x_cluster, p=0.5, training=self.training)
        x_cluster = self.lin2_cluster(x_cluster)

        x_classify = F.relu(self.lin1_classify(out_z))
        x_classify = F.dropout(x_classify, p=0.5, training=self.training)
        x_classify = self.lin2_classify(x_classify)

        return F.log_softmax(x_classify, dim=-1), F.log_softmax(x_cluster, dim=-1), x_hat, out_z

    def __repr__(self):
        return self.__class__.__name__