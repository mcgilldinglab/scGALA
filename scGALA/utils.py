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
from scipy import sparse
from typing import Optional, Tuple


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
        
        # Create edge type indicator: 0 for intra-dataset edges, 1 for inter-dataset edges
        intra_edges_1 = np.zeros(len(edge_1.row), dtype=np.int32)
        intra_edges_2 = np.zeros(len(edge_2.row), dtype=np.int32)
        inter_edges = np.ones(len(MNN_row), dtype=np.int32)
        edge_type = np.concatenate([intra_edges_1, inter_edges, intra_edges_2])
    else:
        row = np.concatenate([edge_1.row,MNN_row,edge_2.row+bias])
        col = np.concatenate([edge_1.col,MNN_col+bias,edge_2.col+bias])
        
        # Create edge type indicator for non-spatial edges
        intra_edges_1 = np.zeros(len(edge_1.row), dtype=np.int32)
        intra_edges_2 = np.zeros(len(edge_2.row), dtype=np.int32)
        inter_edges = np.ones(len(MNN_row), dtype=np.int32)
        edge_type = np.concatenate([intra_edges_1, inter_edges, intra_edges_2])
        
        spatial_row = np.concatenate([spatial_edge_1.row,spatial_edge_2.row+bias])
        spatial_col = np.concatenate([spatial_edge_1.col,spatial_edge_2.col+bias])
        spatial_edge_index = torch.from_numpy(np.array([spatial_row,spatial_col])).contiguous()
        
        # Create edge type indicator for spatial edges (all are intra-dataset)
        spatial_edge_type = np.zeros(len(spatial_row), dtype=np.int32)
        spatial_edge_type = torch.from_numpy(spatial_edge_type)
        
        C1 = cdist(data1.obsm['spatial'],data1.obsm['spatial'],'euclidean')
        C2 = cdist(data2.obsm['spatial'],data2.obsm['spatial'],'euclidean')
        C1 /= C1.max()
        C2 /= C2.max()
    
    edge_index = torch.from_numpy(np.array([row,col])).contiguous()
    edge_type = torch.from_numpy(edge_type)
    
    # Convert to undirected and propagate edge types accordingly
    edge_index_undirected = to_undirected(edge_index).to(torch.int32)
    
    # Propagate edge types to undirected edges (preserve inter-dataset status)
    edge_type_dict = {}
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        edge_type_dict[(src, dst)] = edge_type[i].item()
    
    edge_type_undirected = []
    for i in range(edge_index_undirected.shape[1]):
        src, dst = edge_index_undirected[0, i].item(), edge_index_undirected[1, i].item()
        # Check both directions since we're working with undirected graph
        if (src, dst) in edge_type_dict:
            edge_type_undirected.append(edge_type_dict[(src, dst)])
        elif (dst, src) in edge_type_dict:
            edge_type_undirected.append(edge_type_dict[(dst, src)])
        else:
            # This shouldn't happen, but just in case
            edge_type_undirected.append(0)
    
    edge_type_undirected = torch.tensor(edge_type_undirected, dtype=torch.int32)
    
    x = np.concatenate([data1.X, data2.X],axis=0) # get node features
    x = torch.from_numpy(x).to(torch.float32)
    num_nodes = data1.shape[0] + data2.shape[0]
    
    if not spatial:
        return x, edge_index_undirected, bias, num_nodes, edge_type_undirected
    else:
        # For spatial edge index, also create edge types
        spatial_edge_index_undirected = to_undirected(spatial_edge_index).to(torch.int32)
        spatial_edge_type_undirected = torch.zeros(spatial_edge_index_undirected.shape[1], dtype=torch.int32)
        return x, edge_index_undirected, bias, num_nodes, C1, C2, spatial_edge_index_undirected, edge_type_undirected, spatial_edge_type_undirected

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

def make_alignments(latent: torch.Tensor, mnn1: list, mnn2: list, bias: int, lamb: float, min_value: float = 0, replace=False):
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
        
# Split the data unevenly, making each cell type highly unbalanced while keeping overall train/test ratio
def split_data_unevenly(adata, train_ratio=0.7, group_key='cell_type'):
    train_indices = []
    test_indices = []
    
    # Calculate target number of cells for train set
    target_train_cells = int(adata.n_obs * train_ratio)
    
    groups = adata.obs[group_key].unique()
    
    # Assign extreme split ratios to each group
    group_split_ratios = np.random.uniform(0.1, 0.9, size=len(groups))
    
    for group, split_ratio in zip(groups, group_split_ratios):
        group_indices = np.where(adata.obs[group_key] == group)[0]
        np.random.shuffle(group_indices)
        
        split_point = int(len(group_indices) * split_ratio)
        
        train_indices.extend(group_indices[:split_point])
        test_indices.extend(group_indices[split_point:])
    
    # Adjust to meet overall train ratio
    current_train_ratio = len(train_indices) / (len(train_indices) + len(test_indices))
    
    if current_train_ratio > train_ratio:
        move_to_test = np.random.choice(train_indices, size=int(len(train_indices) - target_train_cells), replace=False)
        train_indices = list(set(train_indices) - set(move_to_test))
        test_indices.extend(move_to_test)
    elif current_train_ratio < train_ratio:
        move_to_train = np.random.choice(test_indices, size=int(target_train_cells - len(train_indices)), replace=False)
        test_indices = list(set(test_indices) - set(move_to_train))
        train_indices.extend(move_to_train)
    
    train_adata = adata[train_indices].copy()
    test_adata = adata[test_indices].copy()
    
    return train_adata, test_adata

# Simulate batch effect
def simulate_batch_effect(data, batch_effect_strength=0.3,noise_strength=0.3):
    if sparse.issparse(data):
        # Convert to dense for batch effect simulation
        data = data.toarray()
    
    # Simulate batch-specific gene expression changes
    n_genes = data.shape[1]
    batch_effect = np.random.normal(1, batch_effect_strength, size=n_genes)
    
    # Apply batch effect
    data_with_batch_effect = data * batch_effect
    
    # Add some random noise
    noise = np.random.normal(0, noise_strength, size=data.shape)
    data_with_batch_effect += noise
    
    # Ensure non-negative values
    data_with_batch_effect = np.clip(data_with_batch_effect, 0, None)
    
    # Convert back to sparse if original was sparse
    if sparse.issparse(data):
        data_with_batch_effect = sparse.csr_matrix(data_with_batch_effect)
    
    return data_with_batch_effect

def compute_anchor_score(adata1, adata2, mnn1,mnn2):
    """
    Calculate integration scores for anchor cells between two AnnData objects
    
    Parameters:
    adata1 (AnnData): First AnnData object
    adata2 (AnnData): Second AnnData object 
    mnn1 (np.array): MNN correspondences for first AnnData object
    mnn2 (np.array): MNN correspondences for second AnnData object
    
    Returns:
    anchor_scores (np.array): Integration scores for anchor cells between the two AnnData objects
    """
    # Get anchor cells
    data1 = adata1.X
    data2 = adata2.X
    bias = adata1.shape[0]
    
    # Calculate each anchor cell's shared neighbors
    dist11 = cdist(data1, data1)
    idx11 = np.argsort(dist11, axis=1)[:, :31]
    dist22 = cdist(data2, data2)
    idx22 = np.argsort(dist22, axis=1)[:, :31]
    dist12 = cdist(data1, data2)
    idx12 = np.argsort(dist12, axis=1)[:, :31]
    idx21 = np.argsort(dist12, axis=0)[:31, :].transpose()
    
    # Calculate shared neighbors
    anchor_scores = np.zeros(len(mnn1))
    for i in range(len(mnn1)):
        all_neighbors1 = np.concatenate([idx11[mnn1[i]], idx12[mnn1[i]]+bias])
        all_neighbors2 = np.concatenate([idx21[mnn2[i]], idx22[mnn2[i]]+bias])
        anchor_scores[i] = len(set(all_neighbors1) & set(all_neighbors2))
    
    # Calculate integration scores
    anchor_scores = (anchor_scores - np.quantile(anchor_scores, 0.01)) / (np.quantile(anchor_scores, 0.90) - np.quantile(anchor_scores, 0.01))
    anchor_scores[anchor_scores < 0] = 0
    anchor_scores[anchor_scores > 1] = 1
    return anchor_scores

class TypedEdgeRemoving:
    r"""Removes a certain percentage of edges for each edge type.

    Args:
        intra_pe (float, optional): Percentage of intra-dataset edges to be removed.
            (default: :obj:`0.3`)
        inter_pe (float, optional): Percentage of inter-dataset edges to be removed.
            (default: :obj:`0.5`)
    """
    def __init__(self, inter_pe: float = 0.5, total_pe: float = 0.3):
        self.inter_pe = inter_pe * total_pe
        self.intra_pe = (1- inter_pe) * total_pe
        
    def __call__(self, x: torch.Tensor, edge_index: torch.Tensor, 
                 edge_type: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        device = edge_index.device
        
        if edge_type is None:
            # If no edge type is provided, use default uniform removal
            num_edges = edge_index.size(1)
            perm = torch.randperm(num_edges, device=device)
            preserve_mask = perm > int(self.intra_pe * num_edges)
            return x, edge_index[:, preserve_mask]
        
        # Process edges by type
        mask = torch.zeros(edge_index.size(1), dtype=torch.bool, device=device)
        
        # Find intra-dataset edges (type 0) and inter-dataset edges (type 1)
        intra_idx = torch.nonzero(edge_type == 0).squeeze()
        inter_idx = torch.nonzero(edge_type == 1).squeeze()
        
        # Apply different removal rates to each type
        if intra_idx.numel() > 0:
            num_intra = intra_idx.numel()
            perm_intra = torch.randperm(num_intra, device=device)
            mask_intra = perm_intra > int(self.intra_pe * num_intra)
            mask.scatter_(0, intra_idx[mask_intra], True)
        
        if inter_idx.numel() > 0:
            num_inter = inter_idx.numel()
            perm_inter = torch.randperm(num_inter, device=device)
            mask_inter = perm_inter > int(self.inter_pe * num_inter)
            mask.scatter_(0, inter_idx[mask_inter], True)
        
        return x, edge_index[:, mask], edge_type[mask]