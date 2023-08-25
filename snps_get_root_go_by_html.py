import requests
from bs4 import BeautifulSoup
import time
from scipy.sparse import csr_matrix
import numpy as np

def save_file(items, connection_path = './data/go_root_connection.txt'):
    file = open(connection_path, 'w')
    for item in items:
        file.write(item + "\n")
    file.close()

def read_go_ids(id_path = "./data/go_ids.txt", link_pre = "https://ctdbase.org/detail.go?type=go&acc=GO%3A"):
    file = open(id_path, "r")
    links = []
    for x in file:
        x = x.replace("GO:","")
        links.append(link_pre+x)
    file.close()
    return links

def load_html_page(id_path, connection_path):
    all_result = []
    links = read_go_ids(id_path=id_path)
    # link = "https://ctdbase.org/detail.go?type=go&acc=GO%3A0048518"
    for link in links:
        time.sleep(30)
        r = requests.get(link)
        soup = BeautifulSoup(r.content, "html.parser")
        for row_index in range(1, 5):
            all_class_topsection = soup.findAll('tr', {'class': 'gridrow%d' % (row_index)})
            if len(all_class_topsection) <= 0:
                break
            for para in all_class_topsection:
                result = para.attrs['id']
                result = result.replace("treeALL.", "")
                result = result.replace("GO", "")
                all_result.append(result)
                print(result)
    save_file(all_result, connection_path=connection_path)

def build_adj(go_adj_row, go_adj_col, go_ids, num_items):
    go_adj_row = np.asarray(go_adj_row)
    go_adj_col = np.asarray(go_adj_col)
    data_val = np.array([1] * len(go_adj_col))
    X = csr_matrix((data_val, (go_adj_row, go_adj_col)), shape=(num_items, num_items))
    adj = X.toarray()
    adj[adj>0]=1
    # degree = np.sum(adj,1)
    # degree_0 = np.sum(adj, 0)
    # print(len(adj), len(go_ids))
    # # print(go_ids)
    # print(degree)
    # print(degree_0)
    # true_index = degree_0==0
    # for i in range(len(true_index)):
    #     if true_index[i]:
    #         print(go_ids[i])
    # # print(adj)
    # root_index = go_ids.index("GO:0008150")
    # print(root_index, degree[root_index])
    return adj


def build_graph_after_loading(connection_path, go_ids, go_adj_row, go_adj_col):
    file = open(connection_path, "r")

    # go_ids = []
    #
    # go_adj_row = []
    # go_adj_col = []

    for x in file:
        go_terms = x.split(".")
        pre_term_index = -1
        for each_term_index in range(len(go_terms)):
            if 2 < each_term_index < len(go_terms)-1:
                continue
            term_id = "GO:" + go_terms[each_term_index].replace('\n','')
            if term_id not in go_ids:
                go_ids.append(term_id)
                # print(term_id)
            term_index = go_ids.index(term_id)

            if pre_term_index >= 0:
                go_adj_col.append(term_index)
                go_adj_row.append(pre_term_index)

            pre_term_index = term_index

        # if len(go_ids)>=10:
        #     break

    file.close()

    adj = build_adj(go_adj_row, go_adj_col, go_ids, len(go_ids))
    return go_ids, adj



if __name__ == "__main__":
    id_path = "./data/go_ids.txt"
    connection_path = './data/go_root_connection.txt'
    # load_html_page(id_path, connection_path)

    go_ids = []
    go_adj_row = []
    go_adj_col = []
    build_graph_after_loading(connection_path, go_ids, go_adj_row, go_adj_col)