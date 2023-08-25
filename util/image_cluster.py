import os
import torch
import scipy.io as sio
from scipy.io import savemat
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
# from yellowbrick.cluster import KElbowVisualizer

def rbf_kernel_torch(X, Y, gamma=0.015): #0.005
    """
    Compute the RBF kernel matrix between X and Y using the specified gamma.

    Args:
        X (torch.Tensor): First input tensor of shape (n_samples_X, n_features).
        Y (torch.Tensor): Second input tensor of shape (n_samples_Y, n_features).
        gamma (float): The gamma parameter of the RBF kernel.

    Returns:
        torch.Tensor: Computed RBF kernel matrix of shape (n_samples_X, n_samples_Y).
    """
    # Compute the pairwise Euclidean distances between X and Y
    distances = torch.cdist(X, Y, p=2)
    # Compute the RBF kernel using the pairwise distances
    kernel_matrix = torch.exp(-gamma * distances ** 2)
    return kernel_matrix


# function returns WSS score for k values from 1 to kmax
def calculate_WSS(points, kmax):
    sse = []
    for k in range(1, kmax + 1):
        kmeans = KMeans(n_clusters=k).fit(points)
        centroids = kmeans.cluster_centers_
        pred_clusters = kmeans.predict(points)
        curr_sse = 0

        # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
        for i in range(len(points)):
            curr_center = centroids[pred_clusters[i]]
            curr_sse += (points[i, 0] - curr_center[0]) ** 2 + (points[i, 1] - curr_center[1]) ** 2

        sse.append(curr_sse)
    return sse

# def run_cluster(data_path='../data/brain_image/', modality_path= 'modality/1/VBM-MRI/'):
#     data_path += modality_path
#     data_path = '../data/snps/data/data_MH/knn_allimg/10/'
#     imgData_mat = sio.loadmat(data_path + 'imgData_mat.mat')
#     imgData_mat_normalized = sio.loadmat(data_path + 'imgData_mat_normalized.mat')
#     BL_DXGrp_label = sio.loadmat(data_path + 'BL_DXGrp_label.mat')
#     imgData_mat = imgData_mat['imgData_mat']
#     imgData_mat_normalized = imgData_mat_normalized['imgData_mat_normalized'][:,:,2]
#     BL_DXGrp_label = BL_DXGrp_label['BL_DXGrp_label']
#
#     d1, d2 = imgData_mat_normalized.shape[0], imgData_mat_normalized.shape[1]
#     # imgData_mat = imgData_mat.reshape((d1, d2))
#     imgData_mat_normalized = imgData_mat_normalized.reshape((d1, d2))
#     BL_DXGrp_label = BL_DXGrp_label.reshape((-1)) -1
#     print(imgData_mat_normalized.shape, BL_DXGrp_label.shape)
#
#     # pca_50 = PCA(n_components=4)
#     # pca_result_50 = pca_50.fit_transform(imgData_mat_normalized)
#     # print('Cumulative explained variation for 50 principal components: {}'.format(
#     #     np.sum(pca_50.explained_variance_ratio_)))
#
#     tsne = TSNE(n_components=2, perplexity=15, init="pca", learning_rate="auto", method="exact", random_state=1000) #, verbose=1, perplexity=40, n_iter=300
#     tsne_results = tsne.fit_transform(imgData_mat_normalized)
#     print(tsne.kl_divergence_)
#
#     # index = np.argsort(BL_DXGrp_label)
#     # BL_DXGrp_label = BL_DXGrp_label[index]
#     # tsne_results = tsne_results[index,:]
#
#     label_plot = []
#     for i in BL_DXGrp_label:
#         if i==0:
#             label_plot.append('HC')
#         elif i==1:
#             label_plot.append('EMCI')
#         elif i==2:
#             label_plot.append('LMCI')
#         elif i==3:
#             label_plot.append('AD')
#
#     d = {'Dimension1': tsne_results[:,0], 'Dimension2': tsne_results[:,1], 'label' : label_plot}
#     dataframe = pd.DataFrame(data=d)
#     plt.figure(1, figsize=(10, 5))
#     g = sns.scatterplot(
#         x="Dimension1", y="Dimension2",
#         hue="label",
#         #palette=sns.color_palette("hls", 4),
#         data=dataframe,
#         legend="full",
#         style = "label",
#         alpha=0.8
#     )
#     # check axes and find which is have legend
#     # leg = g.axes.flat[0].get_legend()
#     # new_title = 'My title'
#     # leg.set_title(new_title)
#     # new_labels = ['label 1', 'label 2', 'label 2']
#     # for t, l in zip(leg.texts, new_labels):
#     #     t.set_text(l)
#     # plt.legend(title='label', labels=['HC', 'EMCI', 'LMCI', 'AD'])
#     # plt.show()
#
#     sse=calculate_WSS(tsne_results, 10)
#     print(sse)
#     plt.figure(2)
#     k=2
#     kmeans = KMeans(n_clusters=k).fit(tsne_results)
#     centroids = kmeans.cluster_centers_
#     pred_clusters = kmeans.predict(tsne_results)
#     d = {'Dimension1': tsne_results[:, 0], 'Dimension2': tsne_results[:, 1], 'label': pred_clusters}
#     dataframe = pd.DataFrame(data=d)
#     plt.figure(2, figsize=(10, 5))
#     g = sns.scatterplot(
#         x="Dimension1", y="Dimension2",
#         hue="label",
#         # palette=sns.color_palette("hls", 4),
#         data=dataframe,
#         legend="full",
#         style="label",
#         alpha=0.9
#     )
#     # plt.plot(sse)
#     plt.show()
#
#     count_number = []
#     for i in range(k):
#         index = pred_clusters==i
#         select_label = BL_DXGrp_label[index]
#         values, counts = np.unique(select_label, return_counts=True)
#         count_number.append(counts)
#     print(count_number)
#
#     print(pred_clusters.shape)




def run_cluster_ADNI874(data_path='../data/snps/data/preprocessing/knn/10/', modality_path= 'modality/1/VBM-MRI/', save_names = 'multimodal'):
    # data_path += modality_path
    # data_path = '../data/snps/data/preprocessing/knn/10/'
    imgData_mat = sio.loadmat(data_path + 'imgData_mat.mat')
    imgData_mat_normalized = sio.loadmat(data_path + 'imgData_mat_normalized.mat')
    BL_DXGrp_label = sio.loadmat(data_path + 'BL_DXGrp_label.mat')
    BL_DXGrp_label = BL_DXGrp_label['BL_DXGrp_label']
    BL_DXGrp_label = BL_DXGrp_label.reshape((-1)) - 1

    if save_names=='multimodal':
        imgData_mat = imgData_mat['imgData_mat'][:, :, :]
        imgData_mat_normalized = imgData_mat_normalized['imgData_mat_normalized'][:, :, :]
        d1, d2, d3 = imgData_mat_normalized.shape
        imgData_mat = imgData_mat.reshape((d1, -1))
        imgData_mat_normalized = imgData_mat_normalized.reshape((d1, -1))
        with open(data_path + 'multimodal_for_similarity.npy', 'wb') as f:
            np.save(f, imgData_mat_normalized)
    else:
        imgData_mat = imgData_mat['imgData_mat'][:,:,2]
        imgData_mat_normalized = imgData_mat_normalized['imgData_mat_normalized'][:,:,2]
        d1, d2 = imgData_mat_normalized.shape[0], imgData_mat_normalized.shape[1]
        imgData_mat = imgData_mat.reshape((d1, d2))
        imgData_mat_normalized = imgData_mat_normalized.reshape((d1, d2))
        with open(data_path + 'pet_for_similarity.npy', 'wb') as f:
            np.save(f, imgData_mat_normalized)

    print(imgData_mat_normalized.shape, BL_DXGrp_label.shape)

    # pca_50 = PCA(n_components=4)
    # pca_result_50 = pca_50.fit_transform(imgData_mat_normalized)
    # print('Cumulative explained variation for 50 principal components: {}'.format(
    #     np.sum(pca_50.explained_variance_ratio_)))



    tsne_results_path = data_path + 'tsne_results_%s.npy'%(save_names)
    if os.path.exists(tsne_results_path):
        tsne_results = np.load(tsne_results_path)
    else:
        ###################### pet perplexity==30, multimodal perplexity==40 ######################
        tsne = TSNE(n_components=2, perplexity=40, init="pca", learning_rate="auto", method="exact", random_state=1000) #, verbose=1, perplexity=30, n_iter=300
        tsne_results = tsne.fit_transform(imgData_mat_normalized)
        print(tsne.kl_divergence_)
        with open(data_path + 'tsne_results_%s.npy'%(save_names), 'wb') as f:
            np.save(f, tsne_results)

    # index = np.argsort(BL_DXGrp_label)
    # BL_DXGrp_label = BL_DXGrp_label[index]
    # tsne_results = tsne_results[index,:]

    label_plot = []
    for i in BL_DXGrp_label:
        if i==0:
            label_plot.append('HC')
        elif i == 1:
            label_plot.append('SMC')
        elif i==2:
            label_plot.append('EMCI')
        elif i==3:
            label_plot.append('LMCI')
        elif i==4:
            label_plot.append('AD')

    d = {'Dimension1': tsne_results[:,0], 'Dimension2': tsne_results[:,1], 'label' : label_plot}
    dataframe = pd.DataFrame(data=d)
    plt.figure(1, figsize=(10, 5))
    g = sns.scatterplot(
        x="Dimension1", y="Dimension2",
        hue="label",
        #palette=sns.color_palette("hls", 4),
        data=dataframe,
        legend="full",
        style = "label",
        alpha=0.6
    )

    # check axes and find which is have legend
    # leg = g.axes.flat[0].get_legend()
    # new_title = 'My title'
    # leg.set_title(new_title)
    # new_labels = ['label 1', 'label 2', 'label 2']
    # for t, l in zip(leg.texts, new_labels):
    #     t.set_text(l)
    # plt.legend(title='label', labels=['HC', 'EMCI', 'LMCI', 'AD'])
    # plt.show()

    sse=calculate_WSS(tsne_results, 10)
    print(sse)
    # plt.figure(2)
    # plt.plot(sse)
    # plt.xticks(range(len(sse)), range(1,len(sse)+1))
    k=2
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=1000).fit(tsne_results) #,max_iter=500
    centroids = kmeans.cluster_centers_
    pred_clusters = kmeans.predict(tsne_results)
    # pred_clusters[pred_clusters==0]=2
    # pred_clusters[pred_clusters==1]=0
    # pred_clusters[pred_clusters==2]=1
    d = {'Dimension1': tsne_results[:, 0], 'Dimension2': tsne_results[:, 1], 'cluster': pred_clusters}
    dataframe = pd.DataFrame(data=d)
    plt.figure(3, figsize=(5, 5))
    g = sns.scatterplot(
        x="Dimension1", y="Dimension2",
        hue="cluster",
        # palette=sns.color_palette("hls", 4),
        data=dataframe,
        legend="full",
        style="cluster",
        alpha=0.9
    )

    g.spines['right'].set_color('none')
    g.spines['top'].set_color('none')
    g.get_legend().remove()
    plt.xlabel('')
    plt.ylabel('')
    plt.savefig('../result_for_visualization/%s_%s.png' % ('image_cluster', save_names), dpi=600, format='png', transparent=True)
    plt.savefig('../result_for_visualization/%s_%s.svg' % ('image_cluster', save_names), dpi=600, format='svg', transparent=True)
    plt.show()

    count_number = []
    for i in range(k):
        index = pred_clusters==i
        select_label = BL_DXGrp_label[index]
        values, counts = np.unique(select_label, return_counts=True)
        count_number.append(counts)
    print(count_number)

    # [array([140,  54, 148,  66,  33], dtype=int64), array([ 59,  29, 122, 105, 118], dtype=int64)]
    #   count_number:  [array([ 50,  23, 117,  97, 111], dtype=int64), array([149,  60, 153,  74,  40], dtype=int64)]
    print(pred_clusters.shape)
    # with open(data_path+'center_%d/clusters_pred_label.npy'%(k), 'wb') as f:
    #     np.save(f, pred_clusters)
    # with open(data_path+'center_%d/clusters_true_label.npy'%(k), 'wb') as f:
    #     np.save(f, BL_DXGrp_label)

    # guas_similarity = visualizeGuassinSimilarity(data_path, imgData_mat_normalized, pred_clusters)

def visualizeGuassinSimilarity(data_path, tsne_results, pred_clusters):
    std_tsne_results = np.reshape(tsne_results,(-1))
    std = np.std(std_tsne_results, 0)
    mean = np.mean(std_tsne_results)
    squared_diff = (std_tsne_results - mean) ** 2
    mean_squared_diff = np.mean(squared_diff)
    print(1/(2*std**2), 1/(2*mean_squared_diff))
    guas_similarity = rbf_kernel(tsne_results, tsne_results, gamma=0.05) #for tsne_result 0.005
    # guas_similarity2 = rbf_kernel_torch(torch.from_numpy(tsne_results), torch.from_numpy(tsne_results))
    cluster_index = np.argsort(pred_clusters)
    guas_similarity = guas_similarity[cluster_index, :]
    guas_similarity = guas_similarity[:, cluster_index]
    print(np.min(guas_similarity), np.max(guas_similarity))

    plt.figure(4, figsize=(6, 6))
    plt.imshow(guas_similarity, cmap='jet', vmin=0, vmax=1)
    plt.colorbar()
    plt.clim(0, 1)
    # plt.savefig('data_brain/statis_ori.png', dpi=600, format='png')
    plt.show()

    mdic = {"guas_similarity": guas_similarity}
    savemat(data_path+"guas_similarity_oripet_0.05.mat", mdic)

    a=1
    return guas_similarity


if __name__ == "__main__":
    run_cluster_ADNI874(save_names = 'multimodal')#multimodal