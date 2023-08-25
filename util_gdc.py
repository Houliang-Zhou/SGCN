import numpy as np
from scipy.linalg import expm
import torch
from scipy.sparse import coo_matrix
from torch_geometric.data import Data, InMemoryDataset

def get_ppr_matrix(
        adj_matrix: np.ndarray,
        alpha: float = 0.05) -> np.ndarray:
    num_nodes = adj_matrix.shape[0]
    A_tilde = adj_matrix #+ np.eye(num_nodes)
    D_tilde = np.diag(1/np.sqrt(A_tilde.sum(axis=1)))
    H = D_tilde @ A_tilde @ D_tilde
    return alpha * np.linalg.inv(np.eye(num_nodes) - (1 - alpha) * H)

def get_heat_matrix(
        adj_matrix: np.ndarray,
        t: float = 5.0) -> np.ndarray:
    num_nodes = adj_matrix.shape[0]
    A_tilde = adj_matrix #+ np.eye(num_nodes)
    D_tilde = np.diag(1/np.sqrt(A_tilde.sum(axis=1)))
    H = D_tilde @ A_tilde @ D_tilde
    return expm(-t * (np.eye(num_nodes) - H))

def get_top_k_matrix(A: np.ndarray, k: int = 5) -> np.ndarray:
    num_nodes = A.shape[0]
    row_idx = np.arange(num_nodes)
    A[A.argsort(axis=0)[:num_nodes - k], row_idx] = 0.
    norm = A.sum(axis=0)
    norm[norm <= 0] = 1 # avoid dividing by zero
    return A/norm

def get_clipped_matrix(A: np.ndarray, eps: float = 0.0001) -> np.ndarray:
    num_nodes = A.shape[0]
    A[A < eps] = 0.
    norm = A.sum(axis=0)
    norm[norm <= 0] = 1 # avoid dividing by zero
    return A/norm

def preprocess_diffusion(data, isPPr = True, isTopK = True, top_k=5, alpha=0.05):
    A = data.A.numpy()

    if isPPr:
        A_diff = get_ppr_matrix(A, alpha=alpha)
    else:
        A_diff = get_heat_matrix(A)

    if isTopK:
        A_res = get_top_k_matrix(A_diff, k=top_k)
    else:
        A_res = get_heat_matrix(A_diff)

    A_coo = coo_matrix(A_res)
    edge_index = torch.from_numpy(np.vstack([A_coo.row, A_coo.col])).long()
    edge_attr = torch.from_numpy(A_coo.data).float()

    data_out = Data(
        x=data.x,
        edge_index=torch.LongTensor(edge_index),
        edge_attr=torch.FloatTensor(edge_attr),
        y=data.y,
        A=data.A,
        A_g = data.A_g,
        gender=data.gender,
        regres_target=data.regres_target,
        adni_type=data.adni_type
    )
    return data_out


def preprocess_diffusion_imgs_snps(data, isPPr = True, isTopK = True, top_k=5, alpha=0.05):
    A = data.A.numpy()

    if isPPr:
        A_diff = get_ppr_matrix(A, alpha=alpha)
    else:
        A_diff = get_heat_matrix(A)

    if isTopK:
        A_res = get_top_k_matrix(A_diff, k=top_k)
    else:
        A_res = get_heat_matrix(A_diff)

    A_coo = coo_matrix(A_res)
    edge_index = torch.from_numpy(np.vstack([A_coo.row, A_coo.col])).long()
    edge_attr = torch.from_numpy(A_coo.data).float()

    data_out = Data(
        x=data.x,
        edge_index=torch.LongTensor(edge_index),
        edge_attr=torch.FloatTensor(edge_attr),
        y=data.y,
        A=data.A,
        snps_feat = data.snps_feat,
        clust_y=data.clust_y,
        sbjID=data.sbjID,
        tsne_fdim=data.tsne_fdim,
        clini_score=data.clini_score,
        demographics=data.demographics
    )
    return data_out