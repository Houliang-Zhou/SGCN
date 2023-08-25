# In[]
import numpy
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import coo_matrix
import numpy as np
import torch
# print(torch.__version__)
from torch_geometric.data import Data
from tqdm import tqdm
import scipy.io as sio
from torch_geometric.data import InMemoryDataset, download_url, extract_zip
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pandas import read_csv
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def loadBrainImg(disease_id=0, class_num=2, isShareAdj=False, isInfo_Score=False, isSeperatedGender=False,
                 selected_gender=1, data_path='./data/brain_image/', adnitype_id=0):
    BL_DXGrp_label = sio.loadmat(data_path+'BL_DXGrp_label.mat')
    corr_data = sio.loadmat(data_path+'corr_data.mat')
    imgData_mat = sio.loadmat(data_path+'imgData_mat.mat')
    imgData_mat_normalized = sio.loadmat(data_path+'imgData_mat_normalized.mat')
    QT_ID = sio.loadmat(data_path+'QT_ID.mat')
    ADNItype = sio.loadmat(data_path+'score_data_nan_ADNItype.mat')
    ADNItype = ADNItype['info_score_subs_ADNItype']
    ADNItype = ADNItype[:, 1].astype(int)
    info_score_subs = sio.loadmat(data_path+'score_data_nan.mat')
    info_score_subs = info_score_subs['info_score_subs']
    BL_DXGrp_label = BL_DXGrp_label['BL_DXGrp_label'] - 1
    corr_data = corr_data['corr_data']
    imgData_mat = imgData_mat['imgData_mat']
    imgData_mat_normalized = imgData_mat_normalized['imgData_mat_normalized']
    # QT_ID = QT_ID['QT_ID_data']

    d1, d2 = imgData_mat_normalized.shape[0], imgData_mat_normalized.shape[1]
    imgData_mat = imgData_mat.reshape((d1, d2, -1))
    imgData_mat_normalized = imgData_mat_normalized.reshape((d1, d2, -1))

    '''
    w or w/o info_score_subs
    -------------------------------------------------------------------------
    '''
    ##using info_score_subs
    if isInfo_Score:
        selected_index = np.intc(info_score_subs[:, 0])
        # sort_selected_index = np.sort(selected_index)
        if isSeperatedGender:
            BL_DXGrp_label = BL_DXGrp_label[selected_index]
            # BL_DXGrp_label = info_score_subs[:,1]
            BL_DXGrp_label = BL_DXGrp_label.reshape((BL_DXGrp_label.shape[0], -1))
            BL_DXGrp_label = np.intc(BL_DXGrp_label)
            regres_targets = np.single(info_score_subs[:, 2:4])
            corr_data = corr_data[selected_index]
            imgData_mat = imgData_mat[selected_index]
            imgData_mat_normalized = imgData_mat_normalized[selected_index]
            gender_info = info_score_subs[:, 4]
            selected_index = gender_info == selected_gender
            BL_DXGrp_label = BL_DXGrp_label[selected_index]
            regres_targets = regres_targets[selected_index]
            corr_data = corr_data[selected_index]
            imgData_mat = imgData_mat[selected_index]
            imgData_mat_normalized = imgData_mat_normalized[selected_index]
        else:
            BL_DXGrp_label = BL_DXGrp_label[selected_index]
            # BL_DXGrp_label = info_score_subs[:,1]
            BL_DXGrp_label = BL_DXGrp_label.reshape((BL_DXGrp_label.shape[0], -1))
            BL_DXGrp_label = np.intc(BL_DXGrp_label)
            gender_info = info_score_subs[:, 4]
            regres_targets = np.single(info_score_subs[:, -2:])
            corr_data = corr_data[selected_index]
            imgData_mat = imgData_mat[selected_index]
            imgData_mat_normalized = imgData_mat_normalized[selected_index]

    ##without using info_score_subs
    else:
        BL_DXGrp_label = BL_DXGrp_label.reshape((BL_DXGrp_label.shape[0], -1))
        BL_DXGrp_label = np.intc(BL_DXGrp_label)
        regres_targets = np.zeros((BL_DXGrp_label.shape[0], 2))

    print("ADNItype: %d, %d, %d" % (np.sum(ADNItype == 0), np.sum(ADNItype == 1), np.sum(ADNItype == 2)))
    values, counts = np.unique(BL_DXGrp_label, return_counts=True)
    print("No. of all labels: Value:", values, "; Count:", counts)
    values, counts = np.unique(BL_DXGrp_label[ADNItype == adnitype_id], return_counts=True)
    print("No. of label under ADNItype 0: Value:", values, "; Count:", counts)
    ADNItype = ADNItype.reshape((ADNItype.shape[0], -1))
    '''
    -------------------------------------------------------------------------
    '''

    dataset = []
    '''
    info_score_subs label: HC=0, MCI=1, AD=2
    -------------------------------------------------------------------------
    '''
    if disease_id == 0:
        select_indices = np.where((BL_DXGrp_label == 0) | (BL_DXGrp_label == 4))[0]
    elif disease_id == 1:
        select_indices = \
        np.where((BL_DXGrp_label == 0) | (BL_DXGrp_label == 2) | (BL_DXGrp_label == 3) | (BL_DXGrp_label == 1))[0]
    elif disease_id == 2:
        select_indices = \
        np.where((BL_DXGrp_label == 4) | (BL_DXGrp_label == 2) | (BL_DXGrp_label == 3) | (BL_DXGrp_label == 1))[0]
    else:
        select_indices = \
            np.where((BL_DXGrp_label == 0) |(BL_DXGrp_label == 4) | (BL_DXGrp_label == 2) | (BL_DXGrp_label == 3) | (BL_DXGrp_label == 1))[0]
    '''
    -------------------------------------------------------------------------
    '''
    BL_DXGrp_label = BL_DXGrp_label[select_indices]
    ADNItype = ADNItype[select_indices]
    imgData_mat = imgData_mat[select_indices]
    corr_data = corr_data[select_indices]
    imgData_mat_normalized = imgData_mat_normalized[select_indices]

    if disease_id == 0:
        BL_DXGrp_label[BL_DXGrp_label > 0] = 1
    elif disease_id == 1:
        BL_DXGrp_label[BL_DXGrp_label > 0] = 1
    elif disease_id == 2:
        BL_DXGrp_label[BL_DXGrp_label == 1] = 0
        BL_DXGrp_label[BL_DXGrp_label == 2] = 0
        BL_DXGrp_label[BL_DXGrp_label == 3] = 0
        BL_DXGrp_label[BL_DXGrp_label == 4] = 1

    normal_sum = np.sum(BL_DXGrp_label == 0)
    mci_sum = np.sum(BL_DXGrp_label == 1)
    ad_sum = np.sum(BL_DXGrp_label == 2)
    print(normal_sum, mci_sum, ad_sum)
    num_of_degrees = []

    share_adj = []
    if isShareAdj:
        for i in range(BL_DXGrp_label.shape[0]):
            share_adj.append(corr_data[i])
        share_adj = np.asarray(share_adj)
        share_adj = share_adj.mean(0)
    for i in range(BL_DXGrp_label.shape[0]):
        X = torch.from_numpy(imgData_mat_normalized[i]).float()
        if isShareAdj:
            A = torch.from_numpy(share_adj).float()
        else:
            A = torch.from_numpy(corr_data[i]).float()
        degree_A = torch.sum(A>0, 1)
        share_adj += [i.item() for i in degree_A]
        regres_target = torch.from_numpy(regres_targets[i]).float()
        if isInfo_Score:
            gender_subject = torch.Tensor([gender_info[i]]).long()
        else:
            gender_subject = torch.Tensor([-1]).long()
        y = torch.Tensor(BL_DXGrp_label[i, :]).long()
        mol_num = torch.Tensor([X.shape[0]])
        adni_type = torch.Tensor(ADNItype[i, :]).long()
        A_coo = coo_matrix(A)
        edge_index = torch.from_numpy(np.vstack([A_coo.row, A_coo.col])).long()
        edge_weight = torch.from_numpy(A_coo.data).float()
        A_g = torch.from_numpy(numpy.ones((50,25))).float()

        data_one = Data(x=X, edge_index=edge_index, edge_attr=edge_weight, y=y,
                        gender=gender_subject,
                        # smiles=smiles_list[i],
                        A=A,
                        A_g=A_g,
                        # atomic_nums=feature_matrices[i],
                        mol_num=mol_num,
                        regres_target=regres_target,
                        adni_type=adni_type)
        dataset.append(data_one)
    print(len(dataset))
    return dataset

def analysis_num_of_degrees(num_of_degrees):
    sns.displot(num_of_degrees)
    plt.show()

def separate_data_adnitype(dataset, disease_id, adnitype_id=0):
    '''

    Args:
        disease_id: 0: HC vs AD; 1: HC vs MCI; 2: MCI vs AD
        adnitype_id: 0

    Returns:

    '''
    test_data = []
    train_data = []
    for data in dataset:
        if data.adni_type == adnitype_id:
            if data.y > 0:
                data.y = torch.Tensor([1]).long()
            test_data.append(data)
        else:
            if disease_id==0:
                if data.y == 0 or data.y == 4:
                    if data.y > 0:
                        data.y = torch.Tensor([1]).long()
                    train_data.append(data)
            elif disease_id==1:
                if data.y == 0 or data.y == 1 or data.y == 2 or data.y == 3:
                    if data.y > 0:
                        data.y = torch.Tensor([1]).long()
                    train_data.append(data)
            elif disease_id==2:
                if data.y == 4 or data.y == 1 or data.y == 2 or data.y == 3:
                    if data.y >= 4:
                        data.y = torch.Tensor([1]).long()
                    else:
                        data.y = torch.Tensor([0]).long()
                    train_data.append(data)
    print('len of train: %d, and test:%d '%(len(train_data),len(test_data)))
    return train_data, test_data

def sparse_matrix(Adj):
    A_coo = coo_matrix(Adj)
    edge_index = torch.from_numpy(np.vstack([A_coo.row, A_coo.col])).long()
    edge_weight = torch.from_numpy(A_coo.data).float()
    return edge_index, edge_weight


class ADNIDataset(InMemoryDataset):
    def __init__(self, root, name, dataset, transform=None, pre_transform=None,
                 pre_filter=None, use_node_attr=False, use_edge_attr=False,
                 cleaned=False):
        self.name = name
        self.cleaned = cleaned
        super(ADNIDataset, self).__init__(root, transform, pre_transform,
                                        pre_filter)
        self.data, self.slices = self.process_trans(dataset)

    @property
    def num_features(self):
        return 3

    @property
    def num_classes(self):
        return 2

    def process_trans(self, dataset):
        new_data_list = []
        for data in tqdm(dataset):
            if self.pre_transform is not None:
                new_data_list.append(self.pre_transform(data))
            else:
                new_data_list.append(data)
        data_list = new_data_list
        self.data, self.slices = self.collate(data_list)
        return self.data, self.slices

if __name__ == "__main__":
    loadBrainImg(disease_id=3, isShareAdj=False, isInfo_Score=True, isSeperatedGender=False, selected_gender=1)
    # loadBrainImg_Snps_CSV(disease_id=0, path = './data/snps/data/%s/', k_inknn = 10)