# from numba import jit, float32, int8, boolean
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from sklearn.preprocessing import normalize
from sklearn.neighbors import kneighbors_graph
import numpy as np
# from sklearn.metrics.pairwise import paired_euclidean_distances
import torch
# from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
# from scipy import stats
# from scipy import sparse
from tqdm import tqdm
from anndata import AnnData
from functools import wraps
from time import time


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r took: %2.4f sec' % 
              (f.__name__, te-ts))
        return result
    return wrap
# @jit((float32[:, :], float32[:, :], int8, int8,boolean, int8), forceobj=True)
# @timing
# def find_mutual_nn(data1, data2, k1, k2,transformed,n_jobs):
#     if not transformed:
#         print('normalizing')
#         data1 = normalize(data1, norm='l2')
#         data2 = normalize(data2, norm='l2')
#     k_index_1 = cKDTree(data1).query(x=data2, k=k1, workers=n_jobs)[1]
#     k_index_2 = cKDTree(data2).query(x=data1, k=k2, workers=n_jobs)[1]
#     mutual_1 = []
#     mutual_2 = []
#     for index_2 in range(data2.shape[0]):
#         for index_1 in k_index_1[index_2]:
#             if index_2 in k_index_2[index_1]:
#                 mutual_1.append(index_1)
#                 mutual_2.append(index_2)
#     return mutual_1, mutual_2

@timing
def find_mutual_nn(data1, data2, k1, k2, transformed, n_jobs):
    if not transformed:
        print('normalizing')
        data1 = normalize(data1, norm='l2')
        data2 = normalize(data2, norm='l2')
        print('normalization done')
    
    # Build KD-Trees and query nearest neighbors
    tree1 = cKDTree(data1)
    tree2 = cKDTree(data2)
    print('tree done')
    k_index_1 = tree1.query(x=data2, k=k1, workers=n_jobs)[1]
    k_index_2 = tree2.query(x=data1, k=k2, workers=n_jobs)[1]
    print('query done')
    # Create pairs from k_index_1
    n2 = data2.shape[0]
    index2s = np.repeat(np.arange(n2), k1)
    index1s = k_index_1.flatten()
    pairs1 = np.column_stack((index1s, index2s))
    
    # Create pairs from k_index_2
    n1 = data1.shape[0]
    index1s_k2 = np.repeat(np.arange(n1), k2)
    index2s_k2 = k_index_2.flatten()
    pairs2 = np.column_stack((index1s_k2, index2s_k2))
    
    # Define a structured data type for efficient comparison
    dtype = np.dtype([('i1', pairs1.dtype), ('i2', pairs1.dtype)])
    structured1 = pairs1.view(dtype).reshape(-1)
    structured2 = pairs2.view(dtype).reshape(-1)
    
    # Find mutual pairs using intersection
    mutual_pairs = np.intersect1d(structured1, structured2)
    
    mutual_1 = mutual_pairs['i1']
    mutual_2 = mutual_pairs['i2']
    
    return mutual_1.tolist(), mutual_2.tolist()
# %%
def get_graph(data1:AnnData,data2:AnnData,mnn1,mnn2,spatial=False):
    # Get graph data
    if not spatial:
        edge_1 = kneighbors_graph(data1.obsm['X_pca'], 20, mode='distance').tocoo()
        edge_2 = kneighbors_graph(data2.obsm['X_pca'], 20, mode='distance').tocoo()
    else:
        edge_1 = kneighbors_graph(data1.obsm['X_pca'], 20, mode='distance').tocoo()
        edge_2 = kneighbors_graph(data2.obsm['X_pca'], 20, mode='distance').tocoo()
        spatial_edge_1 = kneighbors_graph(data1.obsm['spatial'], 5, mode='distance').tocoo()
        spatial_edge_2 = kneighbors_graph(data2.obsm['spatial'], 5, mode='distance').tocoo()
    
    bias = data1.shape[0]
    # total_length = data1.shape[0] + data2.shape[0]
    MNN_row, MNN_col = mnn1,mnn2
    # print(len(MNN_row),len(MNN_col))
    MNN_row = np.array(MNN_row,dtype=np.int32);MNN_col = np.array(MNN_col,dtype=np.int32)
    
    MNN_index = np.array([MNN_row,MNN_col+bias])
    MNN_index = torch.from_numpy(MNN_index)
    
    # Get the concatenated graph
    if not spatial:
        row = np.concatenate([edge_1.row,MNN_row,edge_2.row+bias])
        col = np.concatenate([edge_1.col,MNN_col+bias,edge_2.col+bias])
    else:
        row = np.concatenate([edge_1.row,MNN_row,edge_2.row+bias])
        col = np.concatenate([edge_1.col,MNN_col+bias,edge_2.col+bias])
        spatial_row = np.concatenate([spatial_edge_1.row,spatial_edge_2.row+bias])
        spatial_col = np.concatenate([spatial_edge_1.col,spatial_edge_2.col+bias])
        spatial_edge_index = torch.from_numpy(np.array([spatial_row,spatial_col])).contiguous()
        C1 = cdist(data1.obsm['spatial'],data1.obsm['spatial'],'euclidean')
        C2 = cdist(data2.obsm['spatial'],data2.obsm['spatial'],'euclidean')
        C1 /= C1.max()
        C2 /= C2.max()

    
    edge_index = torch.from_numpy(np.array([row,col])).contiguous()
    
    edge_index = to_undirected(edge_index).to(torch.int32)
    x = np.concatenate([data1.X, data2.X],axis=0) # get node features
    x = torch.from_numpy(x).to(torch.float32)
    num_nodes = data1.shape[0] + data2.shape[0]
    if not spatial:
        return x, edge_index, bias, num_nodes
    else:
        return x, edge_index, bias, num_nodes, C1, C2, spatial_edge_index

def make_alignments_old(latent:torch.Tensor,mnn1:list,mnn2:list,bias:int,lamb:float,min_ppf=0.95,min_percentile:int = 0,min_value:float =0,percent=80,replace=False):
    '''
    make alignment through score-based greedy algorithm with latent space data
    '''
    likelyhood = torch.matmul(latent[:bias], latent[bias:].T).sigmoid()
    likelyhood = np.array(likelyhood.detach().cpu()) # [data1.shape[0],data2.shape[0]]
    marriage_choices = np.zeros_like(likelyhood)

    threshold = min_value
    print(threshold,lamb)
    happiness_scores = np.where(likelyhood>=threshold,likelyhood,0)
    
    # score-based greedy align
    rank_matrix = happiness_scores
    epoch = 0
    complete=0
    aligned_R_count = 0
    aligned_D_count = 0
    patience = 0
    max_epoch = max(min(happiness_scores.shape[0]*2e-2,100),50)
    with tqdm(total=100,position=0, leave=True) as pbar:
        iteration = 0
        while True:
            if iteration % 2 == 1:
                iteration += 1
                available = np.nonzero(np.count_nonzero(rank_matrix,axis=1))[0]
                for i in available:
                    candidate = np.argmax(rank_matrix[i]) # the best candidate
                    diff_d = np.count_nonzero(marriage_choices[i])+np.count_nonzero(marriage_choices[:,candidate])+1 # diff in degree score
                    diff_p = rank_matrix[i,candidate]
                    diff_A = diff_p - lamb*diff_d # diff in total score
                    
                    if diff_A >= 0 :
                        marriage_choices[i,candidate] = rank_matrix[i,candidate] # mark aligned
                        rank_matrix[i,candidate] = 0 # move the successful candidate out of the selection
                    else:# cannot add seat, then try to replace the worst align counterpart
                        if sum(marriage_choices[:,candidate])>0:
                        # find the worst counterpart
                            nonzero_indices = np.nonzero(marriage_choices[:,candidate])
                            nonzero_elements = marriage_choices[:,candidate][nonzero_indices]
                            min_index = np.argmin(nonzero_elements)
                            min_index_in_arr = nonzero_indices[0][min_index] # the worst counterpart
                            
                            diff_d_new = np.count_nonzero(marriage_choices[i])-(np.count_nonzero(marriage_choices[min_index_in_arr])-1) # diff in degree score
                            diff_p_new = rank_matrix[i,candidate] - marriage_choices[min_index_in_arr,candidate] # diff in probability score
                            diff_A_new = diff_p_new - lamb*diff_d_new
                            
                            if diff_A_new > 0 :
                                marriage_choices[i,candidate] = rank_matrix[i,candidate] # mark aligned
                                marriage_choices[min_index_in_arr,candidate] = 0 # remove failed counterpart
                                rank_matrix[i,candidate] = 0 # move the successful candidate out of the selection
                            else:
                                rank_matrix[i,candidate] = 0 # move the failed candidate out of the selection
                        else: rank_matrix[i,candidate] = 0
            else:
                iteration += 1
                available = np.nonzero(np.count_nonzero(rank_matrix,axis=0))[0]
                for i in available:
                    candidate = np.argmax(rank_matrix[:,i]) # the best candidate
                    diff_d = np.count_nonzero(marriage_choices[:,i])+np.count_nonzero(marriage_choices[candidate])+1 # diff in degree score
                    diff_p = rank_matrix[candidate,i]
                    diff_A = diff_p - lamb*diff_d # diff in total score
                    
                    if diff_A >= 0 :
                        marriage_choices[candidate,i] = rank_matrix[candidate,i] # mark aligned
                        rank_matrix[candidate,i] = 0 # move the successful candidate out of the selection
                    else:# cannot add seat, then try to replace the worst align counterpart
                        if sum(marriage_choices[candidate])>0:
                        # find the worst counterpart
                            nonzero_indices = np.nonzero(marriage_choices[candidate])
                            nonzero_elements = marriage_choices[candidate][nonzero_indices]
                            min_index = np.argmin(nonzero_elements)
                            min_index_in_arr = nonzero_indices[0][min_index] # the worst counterpart
                            
                            diff_d_new = np.count_nonzero(marriage_choices[:,i])-(np.count_nonzero(marriage_choices[:,min_index_in_arr])-1) # diff in degree score
                            diff_p_new = rank_matrix[candidate,i] - marriage_choices[candidate,min_index_in_arr] # diff in probability score
                            diff_A_new = diff_p_new - lamb*diff_d_new
                            
                            if diff_A_new > 0 :
                                marriage_choices[candidate,i] = rank_matrix[candidate,i] # mark aligned
                                marriage_choices[candidate,min_index_in_arr] = 0 # remove failed counterpart
                                rank_matrix[candidate,i] = 0 # move the successful candidate out of the selection
                            else:
                                rank_matrix[candidate,i] = 0 # move the failed candidate out of the selection
                        else: rank_matrix[candidate,i] = 0
            epoch+=1
            next_complete = 1-np.count_nonzero(rank_matrix)/rank_matrix.size
            aligned_R = np.count_nonzero(marriage_choices,axis=1)
            aligned_D = np.count_nonzero(marriage_choices,axis=0)
            aligned_R_count_new = np.count_nonzero(aligned_R)
            aligned_D_count_new = np.count_nonzero(aligned_D)
            pbar.set_postfix({'epoch':epoch,'marriage_sum':marriage_choices.sum(),'max_aligned_R':np.max(aligned_R),'aligned_R':aligned_R_count_new,'mean_aligned_R':np.count_nonzero(marriage_choices)/(aligned_R_count_new+1),    'max_aligned_D':np.max(aligned_D),'aligned_D':aligned_D_count_new,'mean_aligned_D':np.count_nonzero(marriage_choices)/(aligned_D_count_new+1)})
            pbar.update((next_complete-complete)*100)
            complete = next_complete
    
            if aligned_R_count_new-aligned_R_count==0 & aligned_D_count_new-aligned_D_count==0:
                patience+=1
            else:
                patience=0
            aligned_R_count=aligned_R_count_new
            aligned_D_count=aligned_D_count_new
            if patience == 3:
                break
            if epoch >= max_epoch:
                break
    if replace:
        return marriage_choices
    # add mnn pairs
    for i,j in zip(mnn1,mnn2):
        marriage_choices[i,j] = 1
    return marriage_choices

def make_alignments(latent: torch.Tensor, mnn1: list, mnn2: list, bias: int, lamb: float, min_ppf=0.95, min_percentile: int = 0, min_value: float = 0, percent=80, replace=False):
    '''
    Optimized version of make alignment through score-based greedy algorithm with latent space data
    '''
    likelyhood = torch.matmul(latent[:bias], latent[bias:].T).sigmoid()
    likelyhood = np.array(likelyhood.detach().cpu())
    threshold = min_value
    print(threshold, lamb)
    
    happiness_scores = np.where(likelyhood >= threshold, likelyhood, 0)
    rank_matrix = happiness_scores.copy()
    marriage_choices = np.zeros_like(happiness_scores)
    
    max_epoch = max(min(int(happiness_scores.shape[0] * 2e-2), 100), 50)
    patience = 0
    complete=0
    
    with tqdm(total=100, position=0, leave=True) as pbar:
        for epoch in range(max_epoch):
            changed = False
            
            # Process rows
            available_rows = np.nonzero(np.count_nonzero(rank_matrix,axis=1))[0]
            row_max_indices = rank_matrix[available_rows].argmax(axis=1)
            candidates = np.column_stack((available_rows, row_max_indices))
            
            for i, j in candidates:
                diff_d = np.count_nonzero(marriage_choices[i]) + np.count_nonzero(marriage_choices[:, j]) + 1
                diff_p = rank_matrix[i, j]
                diff_A = diff_p - lamb * diff_d
                
                if diff_A >= 0:
                    marriage_choices[i, j] = rank_matrix[i, j]
                    rank_matrix[i, j] = 0
                    changed = True
                elif np.count_nonzero(marriage_choices[:, j]) > 0:
                    # Create a mask for positive values in the j-th column
                    positive_mask = marriage_choices[:, j] > 0
                    # Use the mask to filter the positive values
                    positive_values = marriage_choices[positive_mask, j]
                    # Find the index of the minimum positive value
                    worst_index = np.where(positive_mask)[0][positive_values.argmin()]
                    
                    diff_d_new = np.count_nonzero(marriage_choices[i]) - (np.count_nonzero(marriage_choices[worst_index]) - 1)
                    diff_p_new = rank_matrix[i, j] - marriage_choices[worst_index, j]
                    diff_A_new = diff_p_new - lamb * diff_d_new
                    
                    if diff_A_new > 0:
                        marriage_choices[i, j] = rank_matrix[i, j]
                        marriage_choices[worst_index, j] = 0
                        rank_matrix[i, j] = 0
                        changed = True
                    else:
                        rank_matrix[i, j] = 0
                else:
                    rank_matrix[i, j] = 0
            
            # Process columns (similar to rows, but transposed)
            available_cols = np.nonzero(np.count_nonzero(rank_matrix,axis=0))[0]
            col_max_indices = rank_matrix[:, available_cols].argmax(axis=0)
            candidates = np.column_stack((col_max_indices, available_cols))
            
            for i, j in candidates:
                diff_d = np.count_nonzero(marriage_choices[i]) + np.count_nonzero(marriage_choices[:, j]) + 1
                diff_p = rank_matrix[i, j]
                diff_A = diff_p - lamb * diff_d
                
                if diff_A >= 0:
                    marriage_choices[i, j] = rank_matrix[i, j]
                    rank_matrix[i, j] = 0
                    changed = True
                elif np.count_nonzero(marriage_choices[i]) > 0:
                    # Create a mask for positive values
                    positive_mask = marriage_choices[i] > 0
                    # Use the mask to filter the positive values
                    positive_values = marriage_choices[i,positive_mask]
                    # Find the index of the minimum positive value
                    worst_index = np.where(positive_mask)[0][positive_values.argmin()]
                    diff_d_new = np.count_nonzero(marriage_choices[:, j]) - (np.count_nonzero(marriage_choices[:, worst_index]) - 1)
                    diff_p_new = rank_matrix[i, j] - marriage_choices[i, worst_index]
                    diff_A_new = diff_p_new - lamb * diff_d_new
                    
                    if diff_A_new > 0:
                        marriage_choices[i, j] = rank_matrix[i, j]
                        marriage_choices[i, worst_index] = 0
                        rank_matrix[i, j] = 0
                        changed = True
                    else:
                        rank_matrix[i, j] = 0
                else:
                    rank_matrix[i, j] = 0
            
            aligned_R = np.count_nonzero(marriage_choices,axis=1)
            aligned_D = np.count_nonzero(marriage_choices,axis=0)
            aligned_R_count = np.count_nonzero(aligned_R)
            aligned_D_count = np.count_nonzero(aligned_D)
            
            next_complete = 1 - np.count_nonzero(rank_matrix) / rank_matrix.size
            pbar.update((next_complete-complete)*100)
            complete = next_complete
            pbar.set_postfix({
                'epoch': epoch,
                'marriage_sum': marriage_choices.sum(),
                'max_aligned_R': np.max(aligned_R),
                'aligned_R': aligned_R_count,
                'mean_aligned_R': np.count_nonzero(marriage_choices) / (aligned_R_count + 1),
                'max_aligned_D': np.max(aligned_D),
                'aligned_D': aligned_D_count,
                'mean_aligned_D': np.count_nonzero(marriage_choices) / (aligned_D_count + 1)
            })
            
            if not changed:
                patience += 1
                if patience == 3:
                    break
            else:
                patience = 0
    
    if not replace:
        for i, j in zip(mnn1, mnn2):
            marriage_choices[i, j] = 1
    
    return marriage_choices
# # for scDML

import hnswlib
# from annoy import AnnoyIndex
from sklearn.neighbors import NearestNeighbors

def nn_approx(ds1, ds2, names1, names2, knn=50, return_distance=False,metric="cosine",flag="in"):
    dim = ds2.shape[1]
    num_elements = ds2.shape[0]
    if(metric=="euclidean"):
        tree = hnswlib.Index(space="l2", dim=dim)
    elif(metric=="cosine"):
        tree = hnswlib.Index(space="cosine", dim=dim)
    #square loss: 'l2' : d = sum((Ai - Bi) ^ 2)
    #Inner  product 'ip': d = 1.0 - sum(Ai * Bi)
    #Cosine similarity: 'cosine':d = 1.0 - sum(Ai * Bi) / sqrt(sum(Ai * Ai) * sum(Bi * Bi))
    tree.init_index(max_elements=num_elements, ef_construction=200, M=32) # refer to https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md for detail
    tree.set_ef(50)
    tree.add_items(ds2)
    ind, distances = tree.knn_query(ds1, k=knn)
    if(flag=="in"):
        if not return_distance:
            match = set()
            for a, b in zip(range(ds1.shape[0]), ind):
                for b_i in b[1:]:## 
                    match.add((names1[a], names2[b_i]))
            return match
        else:
            match = {}
            for a, b in zip(range(ds1.shape[0]), ind):
                for b_ind, b_i in enumerate(b):
                    match[(names1[a], names2[b_i])] = distances[a, b_ind]  # not sure this is fast
                    # match.add((names1[a], names2[b_i]))
            return match
    else:
        if not return_distance:
            match = set()
            for a, b in zip(range(ds1.shape[0]), ind):
                for b_i in b[0:]:## 
                    match.add((names1[a], names2[b_i]))
            return match
        else:
            match = {}
            for a, b in zip(range(ds1.shape[0]), ind):
                for b_ind, b_i in enumerate(b):
                    match[(names1[a], names2[b_i])] = distances[a, b_ind]  # not sure this is fast
                    # match.add((names1[a], names2[b_i]))
            return match

def nn(ds1, ds2, names1, names2, knn=50, metric_p=2, return_distance=False,metric="cosine",flag="in"):
    # Find nearest neighbors of first dataset.
    if(flag=="in"):
        nn_ = NearestNeighbors(n_neighbors=knn, metric=metric)  # remove self
        nn_.fit(ds2)
        nn_distances, ind = nn_.kneighbors(ds1, return_distance=True)
        if not return_distance:
            match = set()
            for a, b in zip(range(ds1.shape[0]), ind):
                for b_i in b[1:]:
                    match.add((names1[a], names2[b_i]))
            return match
        else:
            match = {}
            for a, b in zip(range(ds1.shape[0]), ind):
                for b_ind, b_i in enumerate(b):
                    match[(names1[a], names2[b_i])] = nn_distances[a, b_ind]  # not sure this is fast
                    # match.add((names1[a], names2[b_i]))
            return match
    else:
        nn_ = NearestNeighbors(n_neighbors=knn, metric=metric)  # remove self
        nn_.fit(ds2)
        nn_distances, ind = nn_.kneighbors(ds1, return_distance=True)
        if not return_distance:
            match = set()
            for a, b in zip(range(ds1.shape[0]), ind):
                for b_i in b[0:]:
                    match.add((names1[a], names2[b_i]))
            return match
        else:
            match = {}
            for a, b in zip(range(ds1.shape[0]), ind):
                for b_ind, b_i in enumerate(b):
                    match[(names1[a], names2[b_i])] = nn_distances[a, b_ind]  # not sure this is fast
                    # match.add((names1[a], names2[b_i]))
            return match