import math
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv, global_add_pool, global_mean_pool, global_sort_pool, global_max_pool
from torch.autograd import Variable
from torch_geometric.utils import to_dense_batch
from pytorch_util import weights_init, gnn_spmm
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn import init
import hyperparameters_baseline as hp


class CNN_Model(torch.nn.Module):
    def __init__(self, class_num=2, hidden_size=64, rois=90):
        super(CNN_Model, self).__init__()
        # explainability
        self.input = None
        self.final_conv_acts = None
        self.final_conv_grads = None
        self.rois = rois
        self.dim1 = self.rois * self.rois
        self.dim2 = 64  # 128  64
        self.dim3 = 16  # 32 16

        # layers
        self.conv1 = torch.nn.Conv2d(1, 4, 3)
        self.conv2 = torch.nn.Conv2d(4, 4, 3)
        self.pool = torch.nn.MaxPool2d(3, stride=3)
        # self.conv2 = GCNConv(H_1, H_2)
        # self.conv3 = GCNConv(H_2, H_3)
        # self.fc1 = torch.nn.Linear(self.rois * H_3, class_num)#29696 51200 59392

        self.fc1 = torch.nn.Linear(324, self.dim2)  # self.rois*H_3  self.dim1    324
        # self.fc1 = torch.nn.Linear(self.dim1, self.dim2)

        # self.bn1 = torch.nn.BatchNorm1d(self.dim2)
        self.fc2 = torch.nn.Linear(self.dim2, self.dim3)
        # self.bn2 = torch.nn.BatchNorm1d(self.dim3)
        self.fc3 = torch.nn.Linear(self.dim3, class_num)
        # self.h1_weights = torch.nn.Linear(59392, hidden_size)
        # self.h2_weights = torch.nn.Linear(hidden_size, class_num)
        weights_init(self)


    def forward(self, data):
        #h0, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        feature = data.A
        feature = feature.view(feature.shape[0]//self.rois, self.rois, self.rois)
        #print(feature.shape)

        feature=feature.unsqueeze(1)
        feature = F.relu(self.conv1(feature))
        feature = self.pool(feature)
        feature = F.relu(self.conv2(feature))
        feature = self.pool(feature)

        #h2 = F.relu(self.conv2(h1, edge_index, edge_weight_prob))
        #h3 = F.relu(self.final_conv_acts)
        feature = feature.view(feature.shape[0], -1)
        # print(h3.shape,data.batch.shape)

        x = feature
        x = F.relu(self.fc1(x))
        # print(x.shape)
        # x = self.bn1(x)
        x = F.dropout(x, p=hp.droupout_prob, training=self.training)
        x = F.relu(self.fc2(x))
        # x = self.bn2(x)
        # x= F.dropout(x, p=0.5, training=self.training)
        logits = F.log_softmax(self.fc3(x), dim=-1)

        return logits

class MLP_Model(torch.nn.Module):
    def __init__(self, class_num=1, hidden_size=64, num_snps=54):
        super(MLP_Model, self).__init__()
        # explainability
        self.input = None
        self.final_conv_acts = None
        self.final_conv_grads = None
        self.num_snps = num_snps
        self.dim1 = num_snps
        self.dim2 = 32  # 128  64
        self.dim3 = 16  # 32 16

        # layers

        self.fc1 = torch.nn.Linear(self.dim1, self.dim2)  # self.rois*H_3  self.dim1    324
        # self.fc1 = torch.nn.Linear(self.dim1, self.dim2)

        # self.bn1 = torch.nn.BatchNorm1d(self.dim2)
        self.fc2 = torch.nn.Linear(self.dim2, self.dim3)
        # self.bn2 = torch.nn.BatchNorm1d(self.dim3)
        self.fc3 = torch.nn.Linear(self.dim3, class_num)
        self.sigmoid = torch.nn.Sigmoid()
        # self.h1_weights = torch.nn.Linear(59392, hidden_size)
        # self.h2_weights = torch.nn.Linear(hidden_size, class_num)
        weights_init(self)


    def forward(self, data):
        #h0, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        # feature = data.A
        # feature = feature.view(feature.shape[0]//self.rois, self.rois, self.rois)

        feature = data.view(data.shape[0], -1)
        # print(h3.shape,data.batch.shape)

        x = feature
        x = F.relu(self.fc1(x))
        # print(x.shape)
        # x = self.bn1(x)
        x = F.dropout(x, p=hp.droupout_prob, training=self.training)
        x = F.relu(self.fc2(x))
        # x = self.bn2(x)
        # x= F.dropout(x, p=0.5, training=self.training)
        logits = self.sigmoid(self.fc3(x))

        return logits