import math
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv, global_add_pool, global_mean_pool, global_sort_pool, global_max_pool
from torch_geometric.utils import to_dense_batch
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn import init
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool, global_add_pool
from torch.nn import Linear
from kernel.guide_go_model import *
from util.image_cluster import rbf_kernel_torch
import torch.nn as nn

class GUIDE_IMGSNP(torch.nn.Module):

    def __init__(self, num_layers, hidden, A_g, A, pool_dim, l_dim, device, *args, hidden_linear=32, rois=90, H_0=3, num_features=3, num_classes=2,
                 isCrossAtten=False, isSoftSimilarity=False, rbf_gamma=0.005, graph_pool=False, isuseFeat4Regr = True, num_regr = 3, model4eachregr = False, isImageOnly = True, isSNPsOnly = False, ifUseGAT=False, **kwargs):
        super(GUIDE_IMGSNP, self).__init__()
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

        n_l=2
        dim_snps_atten = hidden

        self.graph_pool = graph_pool
        # self.lin1 = torch.nn.Linear(90 * num_layers * hidden + linear_input + l_dim + 54, hidden_linear)
        # self.lin1 = torch.nn.Linear(2 * 90 * num_layers * hidden + l_dim + 54, hidden_linear)

        self.latent_dim = 32

        self.lin1 = torch.nn.Linear(self.latent_dim, hidden_linear)
        self.lin1_regr = torch.nn.Linear(self.latent_dim, hidden_linear)
        self.lin2 = Linear(hidden_linear, num_classes)
        self.lin2_regr = torch.nn.Linear(hidden_linear, num_regr)

        # Encoder for nback
        self.encoder_i_N = nn.Sequential(
            nn.Linear(rois*H_0, hidden_linear, bias=False),
            # nn.BatchNorm1d(dd_i_N,track_running_stats=False),
            nn.PReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_linear, self.latent_dim, bias=False),
        )
        # Decoder for Nback
        self.decoder_i_N = nn.Sequential(
            nn.BatchNorm1d(self.latent_dim),
            nn.PReLU(),
            nn.Dropout(0.4),
            nn.Linear(self.latent_dim, hidden_linear, bias=False),
            nn.BatchNorm1d(hidden_linear),
            nn.PReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_linear, rois*H_0, bias=False)
        )
        self.bias_n = nn.ParameterList([nn.Parameter(0.1 * (2 * torch.rand(rois*H_0, 2) - 1))])
        self.prob = [0, 0]

        self.go_network = Gene_ontology_network(A_g, A, 2, n_l, [5, 5], pool_dim, l_dim, device, dim_snps_atten=dim_snps_atten)

        self.batch_norm = torch.nn.BatchNorm1d(num_layers * hidden)


    def reset_parameters(self):
        pass

    def forward(self, data, temperature, device):
        x, edge_index, batch, edge_weight = data.x, data.edge_index, data.batch, data.edge_attr
        snps_feat = data.snps_feat
        x.requires_grad = True
        self.input = x

        x_prob, edge_weight_prob, snps_feat_prob = x, edge_weight, snps_feat

        fill_value = x.min().item() - 1
        batch_x, _ = to_dense_batch(x, batch, fill_value)
        B, N, D = batch_x.size()
        img_out = batch_x.view(B, -1)

        surrogate_ig = []

        # Generate the importance probabilities
        imp_N = F.softmax(self.bias_n[0], dim=1).to(device)
        imp_o_N = imp_N[:, 1]

        x_n = img_out
        if self.training:
            z_N = F.gumbel_softmax(torch.log(imp_N.repeat(x_n.size()[0], 1)), tau=temperature, hard=True)
            x_n_in = x_n * z_N[:, 1].reshape(x_n.size())
        else:
            x_n_in = x_n.clone()

        # Output of the genetic branch
        latent_g, x_hat, _, atten_out = self.go_network(snps_feat_prob, temperature, device)

        surrogate_ig.append(img_out)  # Decoded output for gene scores

        # Latent representation of Nback
        latent_n = self.encoder_i_N(x_n_in)

        # Joint latent representation
        latent = (latent_g + latent_n ) / 2

        # Decoded output for Nback
        surrogate_ig.append(self.decoder_i_N(latent))

        # Importance probabilities.
        prob = [imp_o_N]
        self.prob = [prob[0].detach()]

        latent = latent.reshape((B, -1))
        out_cross = latent
        out_z = latent
        out_lin = latent

        linear_outf = F.relu(self.lin1(latent))
        x = F.dropout(linear_outf, p=0.5, training=self.training)
        x = self.lin2(x)

        our_reg = F.relu(self.lin1_regr(latent))
        our_reg = F.dropout(our_reg, p=0.3, training=self.training)
        our_reg = self.lin2_regr(our_reg)

        return F.log_softmax(x, dim=-1), x_hat, out_z, out_lin, linear_outf, our_reg, surrogate_ig, prob

    def __repr__(self):
        return self.__class__.__name__