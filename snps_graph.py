
import numpy as np
import json
from snps_get_root_go_by_html import build_graph_after_loading
import random
import torch
from torch.utils.data import Dataset
from pandas import read_csv
import pandas as pd
from scipy.sparse import coo_matrix
# from util.GraphVisualization import GraphVisualization

def parse_go_json(file_path):
    f = open(file_path)
    data = json.load(f)

    go_ids = []
    go_subgraph_ids = []
    go_ids_genes = []
    go_ids_fdrs = []
    go_index_ways = []
    go_level_ways = []
    go_root_index_ways = []

    go_adj_row = []
    go_adj_col = []

    for each_way in data['overrepresentation']['group']:
        go_index_perway = []
        go_level_perway = []
        go_root_index_perway = []
        pre_root = -1
        current_root = -1
        term_index = -1
        loop_num = len(each_way['result'])
        if not isinstance(each_way['result'], list):
            loop_num = 1
        for go_term_i in range(loop_num):
            if not isinstance(each_way['result'], list):
                go_term = each_way['result']
            else:
                go_term = each_way['result'][go_term_i]
            term_id = go_term['term']['id']
            term_level = go_term['term']['level']
            term_fdr = go_term['input_list']['fdr']
            term_genes = []
            for each_gene in go_term['input_list']['mapped_id_list']['mapped_id']:
                term_genes.append(each_gene)
            go_ids_genes.append(term_genes)
            if term_id not in go_ids:
                go_ids.append(term_id)
            else:
                # print(term_id)
                pass
            term_index = go_ids.index(term_id)
            go_ids_fdrs.append(term_fdr)

            if term_id not in go_subgraph_ids and go_term_i==0:
                go_subgraph_ids.append(term_id)

            for index_perway in range(len(go_level_perway)-1, -1, -1):
                if term_level > go_level_perway[index_perway]:
                    go_adj_col.append(go_index_perway[index_perway])
                    go_adj_row.append(term_index)
                    break
            if len(go_level_perway)==0 or term_level > go_level_perway[-1]:
                current_root = term_index
            else:
                go_root_index_perway.append(current_root)

                if current_root not in go_subgraph_ids:
                    go_subgraph_ids.append(go_ids[current_root])

                current_root = term_index

            go_index_perway.append(term_index)
            go_level_perway.append(term_level)

        if term_index>=0:
            go_root_index_perway.append(term_index)

            if term_index not in go_subgraph_ids:
                go_subgraph_ids.append(go_ids[term_index])

        go_index_ways.append(go_index_perway)
        go_level_ways.append(go_level_perway)
        go_root_index_ways.append(go_root_index_perway)
        pass
    for root_index_perway in go_root_index_ways:
        for term_index in root_index_perway:
            # print(go_ids[term_index])
            pass
    f.close()
    go_snps, adj, pool_dim, n_l, go_level, new_go_ids, go_ids_genes_list = parse_go_json_subgraph(file_path, go_subgraph_ids)
    return go_snps, adj, pool_dim, n_l, go_level, new_go_ids, go_ids_genes_list

def parse_go_json_subgraph(file_path, go_subgraph_ids):
    f = open(file_path)
    data = json.load(f)

    go_ids = []
    go_ids_genes = []
    go_ids_fdrs = []
    go_index_ways = []
    go_level_ways = []
    go_root_index_ways = []

    go_adj_row = []
    go_adj_col = []

    for each_way in data['overrepresentation']['group']:
        go_index_perway = []
        go_level_perway = []
        go_root_index_perway = []
        pre_root = -1
        current_root = -1
        term_index = -1
        loop_num = len(each_way['result'])
        if not isinstance(each_way['result'], list):
            loop_num = 1
        for go_term_i in range(loop_num):
            if not isinstance(each_way['result'], list):
                go_term = each_way['result']
            else:
                go_term = each_way['result'][go_term_i]
            term_id = go_term['term']['id']
            term_level = go_term['term']['level']
            term_fdr = go_term['input_list']['fdr']

            if term_id not in go_subgraph_ids:
                continue

            term_genes = []
            for each_gene in go_term['input_list']['mapped_id_list']['mapped_id']:
                term_genes.append(each_gene)
            go_ids_genes.append(term_genes)
            if term_id not in go_ids:
                go_ids.append(term_id)
            else:
                # print(term_id)
                pass
            term_index = go_ids.index(term_id)
            go_ids_fdrs.append(term_fdr)

            for index_perway in range(len(go_level_perway)-1, -1, -1):
                if term_level > go_level_perway[index_perway]:
                    go_adj_col.append(go_index_perway[index_perway])
                    go_adj_row.append(term_index)
                    break
            if len(go_level_perway)==0 or term_level > go_level_perway[-1]:
                current_root = term_index
            else:
                go_root_index_perway.append(current_root)
                current_root = term_index
            go_index_perway.append(term_index)
            go_level_perway.append(term_level)
        if term_index>=0:
            go_root_index_perway.append(term_index)
        go_index_ways.append(go_index_perway)
        go_level_ways.append(go_level_perway)
        go_root_index_ways.append(go_root_index_perway)
        pass
    # for root_index_perway in go_root_index_ways:
    #     for term_index in root_index_perway:
    #         print(go_ids[term_index])
    f.close()

    pre_go_ids = [item for item in go_ids]

    print(len(go_ids), len(go_ids_genes))
    connection_path = './data/go_root_connection.txt'
    new_go_ids, adj = build_graph_after_loading(connection_path, go_ids, go_adj_row, go_adj_col)

    go_snps, adj, pool_dim, n_l, go_level, new_go_ids, go_ids_genes_list = build_graph(pre_go_ids, go_ids_genes, new_go_ids, adj)
    return go_snps, adj, pool_dim, n_l, go_level, new_go_ids, go_ids_genes_list

def get_level(adj, current_index, go_level, cur_level):
    num_gos = len(adj)
    child_index = np.arange(num_gos)[adj[current_index,:]>0]
    for each_index in child_index:
        if go_level[each_index]>cur_level+1:
            go_level[each_index]=cur_level+1
        get_level(adj, each_index, go_level, cur_level+1)

def print_pathway(adj, current_index, cur_level, new_go_ids, go_id_pathway):
    num_gos = len(adj)
    child_index = np.arange(num_gos)[adj[current_index,:]>0]
    current_pathway = [item for item in go_id_pathway]
    current_pathway.append(new_go_ids[current_index])
    if len(child_index) <= 0:
        result = ""
        for item_index in range(len(current_pathway)-1):
            result += current_pathway[item_index]+"."
        result += current_pathway[-1]
        print(result)
        return
    for each_index in child_index:
        print_pathway(adj, each_index, cur_level+1, new_go_ids, current_pathway)


def get_genes(adj, current_index, go_ids_genes_map, root_index):
    # if current_index<root_index:
    #     return go_ids_genes_map[current_index]
    num_gos = len(adj)
    child_index = np.arange(num_gos)[adj[current_index, :] > 0]
    gps_current_index = []
    if len(child_index)<=0:
        return go_ids_genes_map[current_index]
    for each_index in child_index:
        result_from_child = []
        result_from_child += get_genes(adj, each_index, go_ids_genes_map, root_index)
        for item in result_from_child:
            if item not in gps_current_index:
                gps_current_index.append(item)
    if current_index>=root_index:
        go_ids_genes_map[current_index] = [item for item in gps_current_index]
    return go_ids_genes_map[current_index]

def preprocess_genes(adj, pre_go_ids, pre_go_ids_genes, new_go_ids, root_index):
    go_ids_genes_map = {}
    for i in range(len(pre_go_ids_genes)):
        go_ids_genes_map[i] = pre_go_ids_genes[i]
    for i in range(len(pre_go_ids_genes), len(new_go_ids)):
        go_ids_genes_map[i] = []
    # get_genes(adj, root_index, go_ids_genes_map, root_index)
    return go_ids_genes_map

def build_go_gene_snps(go_ids_genes_list, root_index, file_path = './data/snps_to_gene.txt'):
    num_go = len(go_ids_genes_list)
    snps_to_genes = []
    file = open(file_path, "r")
    for line in file:
        genes = line.split(";")
        gene_list_per_snps = []
        for gene in genes:
            gene = gene.replace("\n","")
            gene_list_per_snps.append(gene)
        snps_to_genes.append(gene_list_per_snps)
    num_snps = len(snps_to_genes)
    go_snps = np.zeros((num_go, num_snps))
    for i in range(num_go):
        for j in range(num_snps):
            for item in go_ids_genes_list[i]:
                if item in snps_to_genes[j]:
                    go_snps[i, j] = 1
                    break
    for j in range(num_snps):
        go_snps[root_index, j]=1
    return go_snps

def build_graph(pre_go_ids, pre_go_ids_genes, new_go_ids, adj):
    cur_go_genes_length = len(pre_go_ids_genes)
    degree = np.sum(adj, 1)
    degree_0 = np.sum(adj, 0)
    root_index = new_go_ids.index("GO:0008150")
    # print(root_index, degree[root_index], degree_0[root_index])

    go_ids_genes_map = preprocess_genes(adj, pre_go_ids, pre_go_ids_genes, new_go_ids, root_index)
    go_ids_genes_list = []
    for i in range(len(new_go_ids)):
        go_ids_genes_list.append(go_ids_genes_map[i])

    # build graphs based on the level
    num_gos = len(adj)
    go_level = np.ones(num_gos)*np.inf #0  np.inf
    go_level[root_index] = 0
    get_level(adj, root_index, go_level, cur_level=0)
    print(go_level)

    go_id_pathway = []
    # print_pathway(adj, root_index, 0, new_go_ids, go_id_pathway)

    sort_index = np.argsort(-go_level)
    go_level = go_level[sort_index]
    new_go_ids = [new_go_ids[item_index] for item_index in sort_index]
    go_ids_genes_list = [go_ids_genes_list[item_index] for item_index in sort_index]
    adj = adj[sort_index, :]
    adj = adj[:, sort_index]

    degree = np.sum(adj, 1)
    degree_0 = np.sum(adj, 0)
    root_index = new_go_ids.index("GO:0008150")
    print(root_index, degree[root_index], degree_0[root_index])

    n_l = 4
    pool_dim = []
    for i in range(4,-1,-1):
        pool_dim.append(np.sum(go_level==i))
    pool_dim = [pool_dim]

    go_snps = build_go_gene_snps(go_ids_genes_list, root_index)

    return go_snps, adj, pool_dim, n_l, go_level, new_go_ids, go_ids_genes_list

class SnpsDataset(Dataset):
    def __init__(self, disease_id=0, training_index=None, test_index=None, isTrain=True, path = './data/snps/data/%s/', isAllData = False):
        file_path = path
        if disease_id==0:
            file_path = file_path % ('data_AH')
        elif disease_id==1:
            file_path = file_path % ('data_MH')
        else:
            file_path = file_path % ('data_AM')
        data = read_csv(file_path+'snp.csv')
        label = read_csv(file_path+'dia.csv')
        data = data.values/10
        label = label.values
        print('loading number of data: ', np.sum(label<=0), np.sum(label>0))
        # print(data)
        # print(label)
        # self.data = torch.tensor(data.astype(float)).float()
        # self.label = torch.tensor(label.reshape(-1)).long()
        if isAllData:
            self.data = torch.from_numpy(data).float()
            self.target = torch.from_numpy(label).float()
        elif isTrain:
            self.data = torch.from_numpy(data[training_index]).float()
            self.target = torch.from_numpy(label[training_index]).float()
        else:
            self.data = torch.from_numpy(data[test_index]).float()
            self.target = torch.from_numpy(label[test_index]).float()
        self.length = self.data.shape[0]

    def __len__(self):
        return self.length

    def get_labels(self):
        return self.target

    def __getitem__(self, index):
        return self.data[index], self.target[index]

# def visualize_snps_graph(adj, go_level, new_go_ids, go_ids_genes_list):
#
#     sort_index = np.argsort(go_level)
#     go_level_copy = go_level[sort_index]
#     new_go_ids_copy = [new_go_ids[item] for item in sort_index]
#     go_ids_genes_list_copy = [go_ids_genes_list[item] for item in sort_index]
#     adj = adj[sort_index, :]
#     adj = adj[:, sort_index]
#
#     pos_node = {}
#     pre_level = 0
#     cur_level = 0
#     cur_index = 0
#     for i in range(len(go_level_copy)):
#         cur_level = go_level_copy[i]
#         if pre_level != cur_level:
#             cur_index = 0
#         elif i > 0:
#             cur_index += 1
#         diff1 = random.random()
#         diff2 = random.uniform(0, 0.13)
#         if diff1 < 0.5:
#             pos_node[i] = [cur_index, -cur_level + diff2]
#         else:
#             pos_node[i] = [cur_index, -cur_level - diff2]
#         pre_level = cur_level
#
#     a=np.sum(adj,0)
#     b=np.sum(adj,1)
#
#     select_num_node = len(adj)
#
#     adj = adj[:select_num_node, :]
#     adj = adj[:, :select_num_node]
#
#     adj_coo = coo_matrix(adj)
#     G = GraphVisualization()
#     for i in range(len(adj_coo.row)):
#         G.addEdge(adj_coo.col[i], adj_coo.row[i])
#     # G.visualize(pos_node)
#
#     filepath = './result_snps/snps_graph.csv'
#     df = pd.DataFrame({'GO_ID': new_go_ids_copy, 'Genes': go_ids_genes_list_copy})
#     df.to_csv(filepath)

if __name__ == "__main__":
    path = './data/snps/analysis.json'
    go_snps, adj, pool_dim, n_l, go_level, new_go_ids, go_ids_genes_list = parse_go_json(path)
    a=1

    # visualize_snps_graph(adj, go_level, new_go_ids, go_ids_genes_list)

    #
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # # A = np.random.randint(2, size=(20, 20))
    # A = gpu(torch.tensor(adj).float().t().to_sparse().coalesce())
    # # A_g = np.random.randint(2, size=(20, 54))
    # A_g = gpu(torch.tensor(go_snps).float().to_sparse().coalesce())
    #
    # x = gpu_t(np.random.randn(30, 54))/2
    # pool_dim = np.asarray(pool_dim).tolist()
    # l_dim = 16
    # t = gpu_ts(0.1)
    # net = Gene_ontology_network(A_g, A, 2, 3, [5, 5, 5], pool_dim, l_dim)
    # net = net.to(device)
    # latent, x_hat, prob = net(x, t)
    # print(latent.shape, x_hat.shape)

    # SnpsDataset()
