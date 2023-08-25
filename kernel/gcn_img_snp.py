import math
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv, global_add_pool, global_mean_pool, global_sort_pool, global_max_pool
from torch_geometric.utils import to_dense_batch
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn import init
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool, global_add_pool
from torch.nn import Linear
from kernel.go_model import *
from util.image_cluster import rbf_kernel_torch

class GCN_IMGSNP(torch.nn.Module):

    def __init__(self, num_layers, hidden, A_g, A, pool_dim, l_dim, device, *args, hidden_linear=64, rois=90, H_0=3, num_features=3, num_classes=2,
                 isCrossAtten=False, isSoftSimilarity=False, rbf_gamma=0.005, graph_pool=False, isuseFeat4Regr = True, num_regr = 4, model4eachregr = False, isImageOnly = True, isSNPsOnly = False, ifUseGAT=False, **kwargs):
        super(GCN_IMGSNP, self).__init__()
        self.device=device
        self.isCrossAtten=isCrossAtten
        self.isSoftSimilarity=isSoftSimilarity
        self.rbf_gamma = rbf_gamma
        self.isuseFeat4Regr=isuseFeat4Regr
        self.model4eachregr=model4eachregr
        self.isImageOnly=isImageOnly
        self.isSNPsOnly=isSNPsOnly
        self.num_regr=num_regr
        self.input = None
        self.final_conv_acts = None
        self.final_conv_grads = None
        self.rois = rois
        self.prob_dim = H_0
        self.conv1 = GCNConv(num_features, hidden) if not ifUseGAT else GATConv(num_features, hidden, edge_dim=1)
        self.convs = torch.nn.ModuleList()
        n_l=2
        dim_snps_atten = hidden
        if isCrossAtten:
            for i in range(num_layers - 2):
                self.convs.append(GCNConv(hidden, hidden) if not ifUseGAT else GATConv(hidden, hidden, edge_dim=1))
                dim_snps_atten+=hidden
            self.convs.append(GCNConv(hidden, hidden) if not ifUseGAT else GATConv(hidden, hidden, edge_dim=1))
            dim_snps_atten += hidden
            self.pool = pool_dim[0]
            linear_input = (sum(self.pool) - sum(self.pool[0:n_l]))*dim_snps_atten
            self.multihead_attn = torch.nn.MultiheadAttention(dim_snps_atten, 2, batch_first=True)
        else:
            for i in range(num_layers - 1):
                self.convs.append(GCNConv(hidden, hidden) if not ifUseGAT else GATConv(hidden, hidden, edge_dim=1))
            linear_input = 90 * num_layers * hidden + l_dim
        self.graph_pool = graph_pool
        # self.lin1 = torch.nn.Linear(90 * num_layers * hidden + linear_input + l_dim + 54, hidden_linear)
        # self.lin1 = torch.nn.Linear(2 * 90 * num_layers * hidden + l_dim + 54, hidden_linear)
        if self.graph_pool:
            # print(num_layers, hidden, l_dim)
            self.lin1 = torch.nn.Linear(3 * num_layers * hidden + l_dim, hidden_linear)
            self.lin1_regr = torch.nn.Linear(3 * num_layers * hidden + l_dim, hidden_linear)
        else:
            if isImageOnly:
                self.lin1 = torch.nn.Linear(90 * num_layers * hidden, hidden_linear)
            elif isSNPsOnly:
                self.lin1 = torch.nn.Linear(l_dim + 54, hidden_linear)
            else:
                self.lin1 = torch.nn.Linear(90 * num_layers * hidden + l_dim, hidden_linear)
            if isuseFeat4Regr:
                if model4eachregr:
                    self.lin1_regr = []
                    for i in range(num_regr):
                        self.lin1_regr.append(torch.nn.Linear(90 * num_layers * hidden + l_dim + 90 * H_0, hidden_linear).to(device))
                else:
                    if isImageOnly:
                        self.lin1_regr = torch.nn.Linear(90 * num_layers * hidden + 90 * H_0, hidden_linear)
                    elif isSNPsOnly:
                        self.lin1_regr = torch.nn.Linear(l_dim + 54, hidden_linear)
                    else:
                        self.lin1_regr = torch.nn.Linear(90 * num_layers * hidden + l_dim + 90 * H_0, hidden_linear)
            else:
                if isImageOnly:
                    self.lin1_regr = torch.nn.Linear(90 * num_layers * hidden, hidden_linear)
                elif isSNPsOnly:
                    self.lin1_regr = torch.nn.Linear(l_dim + 54, hidden_linear)
                else:
                    self.lin1_regr = torch.nn.Linear(90 * num_layers * hidden + l_dim, hidden_linear)
        self.lin2 = Linear(hidden_linear, num_classes)
        if model4eachregr:
            self.lin2_regr = []
            for i in range(num_regr):
                self.lin2_regr.append(torch.nn.Linear(hidden_linear, 1).to(device))
        else:
            self.lin2_regr = torch.nn.Linear(hidden_linear, num_regr)

        self.batch_norm_1d = torch.nn.BatchNorm1d(num_features= 90 * num_layers * hidden + l_dim)

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
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        if self.model4eachregr:
            for i in range(self.num_regr):
                self.lin1_regr[i].reset_parameters()
                self.lin2_regr[i].reset_parameters()
        else:
            self.lin1_regr.reset_parameters()
            self.lin2_regr.reset_parameters()

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

    def consist_loss(self, s, tsne_result=None):
        if len(s) == 0:
            return 0
        # s = torch.sigmoid(s)
        if self.isSoftSimilarity and tsne_result is not None:
            W = rbf_kernel_torch(tsne_result, tsne_result, gamma=self.rbf_gamma)
        else:
            W = torch.ones(s.shape[0], s.shape[0])
        W = W.to(self.device)
        D = torch.eye(s.shape[0]).to(self.device) * torch.sum(W, dim=1)
        L = D - W
        L = L
        cluster_loss = torch.trace(torch.transpose(s, 0, 1) @ L @ s) / (s.shape[0] * s.shape[0])
        return cluster_loss

    def OrthogonalConstraint(self, w):
        # Calculate the orthogonality penalty loss.
        weight = w /w.norm(dim=1)[:, None]
        temp_matmul = weight.T @ weight
        penalty = torch.norm(temp_matmul - torch.eye(weight.shape[1]).to(self.device)) ** 2
        # Add the orthogonality penalty loss to the original loss.
        loss = penalty / (weight.shape[0] * weight.shape[0])
        return loss

    def forward(self, data, temperature, device):
        x, edge_index, batch, edge_weight = data.x, data.edge_index, data.batch, data.edge_attr
        snps_feat = data.snps_feat
        x.requires_grad = True
        self.input = x

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

        if self.graph_pool:
            mean_z = global_mean_pool(x, batch)
            max_z = global_max_pool(x, batch)
            add_z = global_add_pool(x, batch)
            xs = [mean_z]+[max_z]+[add_z]
            img_out = torch.cat(xs, dim=1)

        latent, x_hat, _, atten_out = self.go_network(snps_feat_prob, temperature, device)
        # x_hat = snps_feat

        if self.isCrossAtten:
            attn_output, attn_output_weights = self.multihead_attn(batch_x, atten_out, atten_out)
            attn_output = F.relu(attn_output)
            out_cross = attn_output #.reshape((attn_output.shape[0], -1))
        else:
            out_cross = torch.cat((img_out, latent), -1)

        # if self.isCrossAtten:
        #     attn_output, attn_output_weights = self.multihead_attn(atten_out, batch_x, batch_x)
        #     x = attn_output.reshape((attn_output.shape[0], -1))
        # else:
        #     x = torch.cat((img_out, latent), -1)

        # x = torch.cat((img_out, x, F.relu(snps_feat), latent), -1)
        if self.graph_pool:
            out_cross = out_cross.reshape((out_cross.shape[0]*out_cross.shape[1],out_cross.shape[2]))
            mean_z = global_mean_pool(out_cross, batch)
            max_z = global_max_pool(out_cross, batch)
            add_z = global_add_pool(out_cross, batch)
            xs = [mean_z]+[max_z]+[add_z]
            out_cross = torch.cat(xs, dim=1)
        else:
            out_cross = out_cross.reshape((B, -1))
        x_prob = x_prob.reshape((B, -1))
        # print(torch.max(img_out),torch.max(x),torch.max(latent),torch.max(snps_feat_prob))

        # out_z = torch.cat((img_out, x, latent, snps_feat_prob), -1)

        if self.isImageOnly:
            out_z = img_out
            out_lin = out_z
        elif self.isSNPsOnly:
            out_z = latent
            out_lin = torch.cat((snps_feat_prob, latent), -1)
        else:
            out_z = (img_out + out_cross) / 2
            out_lin = torch.cat((out_z, latent), -1)

        # out_z = self.batch_norm_1d(out_z)

        # x = torch.permute(x, (0,2,1))
        # batch_x = torch.permute(batch_x, (0,2,1))
        # x = self.batch_norm(x+batch_x)
        # x = x.view(B, -1)
        # x = torch.cat((latent, x), -1)

        linear_outf = F.relu(self.lin1(out_lin))
        x = F.dropout(linear_outf, p=0.5, training=self.training)
        x = self.lin2(x)

        if self.isuseFeat4Regr:
            fill_value = data.x.min().item() - 1
            batch_x, _ = to_dense_batch(data.x, batch, fill_value)
            B, N, D = batch_x.size()
            batch_x = batch_x * self.prob #* torch.sigmoid(self.prob)
            # batch_x = batch_x[:,-2:,:]
            img_feat = batch_x.view(B, -1)
            feat4regr = torch.cat((out_lin, img_feat), -1)
            if self.model4eachregr:
                our_reg_all = []
                for i in range(self.num_regr):
                    our_reg = F.relu(self.lin1_regr[i](feat4regr))
                    our_reg = F.dropout(our_reg, p=0.3, training=self.training)
                    our_reg = self.lin2_regr[i](our_reg)
                    our_reg_all.append(our_reg)
                our_reg = torch.cat(our_reg_all,-1)
            else:
                our_reg = F.relu(self.lin1_regr(feat4regr))
                our_reg = F.dropout(our_reg, p=0.3, training=self.training)
                our_reg = self.lin2_regr(our_reg)
        else:
            our_reg = F.relu(self.lin1_regr(out_lin))
            our_reg = F.dropout(our_reg, p=0.3, training=self.training)
            our_reg = self.lin2_regr(our_reg)

        # linear_outf = torch.zeros_like(linear_outf).to(device)
        # x = torch.zeros_like(x).to(device)

        return F.log_softmax(x, dim=-1), x_hat, out_z, out_lin, linear_outf, our_reg

    def __repr__(self):
        return self.__class__.__name__