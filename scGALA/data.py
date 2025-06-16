# For Cell Alignment
from anndata import AnnData
from torch.utils.data import Dataset,DataLoader
import torch_geometric.data as pyg_data
from .utils import get_graph
import lightning as L
import scanpy as sc

class FullBatchDataset(Dataset):
    def __init__(self,adata1:AnnData,adata2:AnnData,mnn1,mnn2,length,spatial=False):
        super().__init__()
        self.data = get_graph(data1=adata1,data2=adata2,mnn1=mnn1,mnn2=mnn2,spatial=spatial)
        self.length = length

    def __getitem__(self, index):
        return self.data

    def __len__(self):
        return self.length

class MyDataModule(L.LightningDataModule):
    def __init__(self, adata1:AnnData,adata2:AnnData,mnn1,mnn2,spatial=False):
        super().__init__()
        self.train_dataset = FullBatchDataset(adata1,adata2,mnn1,mnn2,length=20,spatial=spatial)
        self.val_dataset = FullBatchDataset(adata1,adata2,mnn1,mnn2,length=1,spatial=spatial)
    def setup(self, stage):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        pass
    def train_dataloader(self):
        return DataLoader(self.train_dataset)

    def val_dataloader(self):
        return DataLoader(self.val_dataset)

    # def test_dataloader(self):
    #     return DataLoader(self.dataset)
    
    def predict_dataloader(self):
        return DataLoader(self.val_dataset)

    def teardown(self,stage):
        # clean up state after the trainer stops, delete files...
        # called on every process in DDP
        ...
        
# For Multiomics Generation
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
import pandas as pd

class DataProcessor:
    def __init__(self, n_neighbors=10):
        self.n_neighbors = n_neighbors
    
    def load_data(self, rna, atac2rna, anchor_path):
        self.rna_adata = rna
        self.atac_adata = atac2rna
        # anchors: [(rna_idx, atac_idx, score), ...]
        self.anchors = pd.read_csv(anchor_path).values.tolist()
        
    def construct_knn_graph(self, X):
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors)
        nbrs.fit(X)
        distances, indices = nbrs.kneighbors(X)
        
        edge_index = []
        edge_attr = []
        for i, (dist, idx) in enumerate(zip(distances, indices)):
            for d, j in zip(dist, idx):
                edge_index.append([i, j])
                edge_attr.append(np.exp(-d))  # similarity-based edge weight
        
        return torch.tensor(edge_index).t().contiguous(), torch.tensor(edge_attr)
    
    def prepare_data(self):
        X = torch.FloatTensor(self.atac_adata.X.toarray())
        Y = torch.FloatTensor(self.rna_adata.X.toarray())
        
        # Construct KNN graph for ATAC cells
        edge_index, edge_attr = self.construct_knn_graph(X.numpy())
        
        # Create anchor mapping tensors
        rna_idx = []
        atac_idx = []
        anchor_weights = []
        for rna_i, atac_i, score in self.anchors:
            rna_idx.append(int(rna_i)-1)
            atac_idx.append(int(atac_i)-1)
            anchor_weights.append(score)
            
        anchor_map = {
            'rna_idx': torch.LongTensor(rna_idx),
            'atac_idx': torch.LongTensor(atac_idx),
            'weights': torch.FloatTensor(anchor_weights)
        }
        
        return Data(x=X, y=Y, edge_index=edge_index, 
                   edge_attr=edge_attr, anchor_map=anchor_map)

# For Spatial Imputation
from torch_geometric.utils import to_undirected
from sklearn.neighbors import kneighbors_graph
import time

def reorder_adata_genes(adata, target_adata,genes=None,n_matching_genes=None):
    """
    Reorder genes in adata to match the order of target_adata, 
    with additional genes from adata appended at the end.

    Parameters:
    -----------
    adata : AnnData
        The AnnData object with more genes to be reordered.
    target_adata : AnnData
        The AnnData object with fewer genes, used as the reference for reordering.

    Returns:
    --------
    AnnData
        A new AnnData object with reordered genes.
    """
    if genes is None:
        # Get the list of genes from both AnnData objects
        adata_genes = adata.var_names.tolist()
        target_genes = target_adata.var_names.tolist()

        # Find common genes and genes only in adata
        adata_gene_set = set(adata_genes)
        target_gene_set = set(target_genes)
        common_genes = list(target_gene_set.intersection(adata_gene_set))
        only_adata_genes = list(adata_gene_set.difference(target_gene_set))
        # Create the new gene order
        new_gene_order = common_genes + only_adata_genes
    else:
        common_genes = genes[:n_matching_genes]
        only_adata_genes = genes[n_matching_genes:]
        new_gene_order = genes
    target_adata_common = target_adata[:,common_genes]
    # Reorder the adata object
    reordered_adata = adata[:, new_gene_order]

    return reordered_adata,target_adata_common,len(common_genes)
class MyDataModule_OneStage(L.LightningDataModule):
    def __init__(self, adata: AnnData, target_adata: AnnData,k=20,save=True,mnn1=None,mnn2=None):
        super().__init__()
        start_time = time.time()
        reordered_adata,target_adata_common, n_matching_genes = reorder_adata_genes(adata, target_adata)
        end_time = time.time()
        print(f"Time taken to reorder adata genes: {end_time - start_time:.4f} seconds")
        print(f'{n_matching_genes} matching genes and {adata.shape[1]-n_matching_genes} only in adata')
        self.n_matching_genes = n_matching_genes
        self.x, self.edge_index, self.bias, self.num_nodes = get_graph_spatial(reordered_adata,target_adata_common,mnn1,mnn2,k)
        if save:
            with open('var_names_one_stage.txt', 'w') as f:
                f.write(f'{n_matching_genes}\n')
                for var_name in reordered_adata.var_names:
                    f.write(f"{var_name}\n")
            print('var_names saved in var_names_one_stage.txt')
        self.data = Data(x=self.x, edge_index=self.edge_index, bias=self.bias, num_nodes=self.num_nodes)
    def setup(self, stage):
        # No need for train/val split in this case
        pass

    def train_dataloader(self):
        return pyg_data.DataLoader([self.data],batch_size=1)

    def val_dataloader(self):
        return pyg_data.DataLoader([self.data],batch_size=1)
    
    def predict_dataloader(self):
        return pyg_data.DataLoader([self.data],batch_size=1)

    def teardown(self, stage):
        # Clean up state after the trainer stops, delete files...
        if stage == 'predict':
            del self.data

    
def get_graph_spatial(data1:AnnData,data2:AnnData,mnn1,mnn2,k=20):
    # Get graph data
    edge_1 = kneighbors_graph(data1.obsm['X_pca'], k, mode='distance').tocoo()
    
    if 'spatial' not in data2.obsm:
        print("data2 does not have spatial coordinates, using PCA instead.")
        edge_2 = kneighbors_graph(data2.obsm['X_pca'], k, mode='distance').tocoo()
    else:
        edge_2_pca = kneighbors_graph(data2.obsm['spatial'], int(k/2), mode='distance')
        edge_2_spatial = kneighbors_graph(data2.obsm['spatial'], int(k/2), mode='distance')
        # Combine the two edges in coo format
        edge_2 = edge_2_pca + edge_2_spatial
        edge_2 = edge_2.tocoo()
    
    # find the index of mnn1 and mnn2 in data1 and data2 if mnn1 and mnn2 are obs_names
    if isinstance(mnn1[0],str):
        mnn1_index = data1.obs_names.get_indexer(mnn1)
    else:
        mnn1_index = mnn1
    if isinstance(mnn2[0],str):
        mnn2_index = data2.obs_names.get_indexer(mnn2)
    else:
        mnn2_index = mnn2
    
    ## make data2 the feature size as data1, fill the rest with zeros
    data2_new = AnnData(X=np.zeros((data2.shape[0],data1.shape[1]),dtype=np.float32))
    # print(f"data1 shape: {data1.shape}, data2 shape: {data2.shape}, data2_new shape: {data2_new.shape}")
    data2_new.X[:,:data2.n_vars] = data2.X.toarray() if hasattr(data2.X, 'toarray') else data2.X
    # print(f"data2_new after filling: {data2_new}")
    bias = data1.shape[0]
    # total_length = data1.shape[0] + data2.shape[0]
    MNN_row, MNN_col = mnn1_index,mnn2_index
    # print(len(MNN_row),len(MNN_col))
    MNN_row = np.array(MNN_row,dtype=np.int32);MNN_col = np.array(MNN_col,dtype=np.int32)
    
    MNN_index = np.array([MNN_row,MNN_col+bias])
    MNN_index = torch.from_numpy(MNN_index)
    
    # Get the concatenated graph
    row = np.concatenate([edge_1.row,MNN_row,edge_2.row+bias])
    col = np.concatenate([edge_1.col,MNN_col+bias,edge_2.col+bias])

    edge_index = torch.from_numpy(np.array([row,col])).contiguous()
    
    edge_index = to_undirected(edge_index).to(torch.int32)
    # print(data1.X.shape,data2_new.X.shape)
    data1_x = data1.X.toarray() if hasattr(data1.X, 'toarray') else data1.X
    data2_new_x = data2_new.X.toarray() if hasattr(data2_new.X, 'toarray') else data2_new.X
    # Concatenate the node features
    x = np.concatenate([data1_x, data2_new_x],axis=0) # get node features
    x = torch.from_numpy(x).to(torch.float32)
    num_nodes = data1.shape[0] + data2.shape[0]
    return x, edge_index, bias, num_nodes

# For Improved Spatial Imputation with scGALA: Two-Stage Data Module
import os
import pickle

def select_centroid_patient(adata, method='pca', patient_key='patient'):
    """
    Select the centroid patient/sample based on UMAP or PCA coordinates.
    Returns the patient/sample id.
    """
    if method == 'umap':
        if 'X_umap' not in adata.obsm:
            import scanpy as sc
            sc.tl.umap(adata)
        coords = adata.obsm['X_umap']
    else:
        if 'X_pca' not in adata.obsm:
            import scanpy as sc
            sc.pp.pca(adata)
        coords = adata.obsm['X_pca']
    patients = adata.obs[patient_key].unique()
    centroids = []
    for p in patients:
        idx = adata.obs[patient_key] == p
        centroids.append(coords[idx].mean(axis=0))
    centroids = np.vstack(centroids)
    overall_centroid = coords.mean(axis=0)
    dists = np.linalg.norm(centroids - overall_centroid, axis=1)
    centroid_patient = patients[np.argmin(dists)]
    return centroid_patient

def construct_and_save_intersample_edges(adata, save_path, k=20, centroid_patient=None, devices=[0], force_recompute=False, patient_key='patient', centroid_method='pca',spatial=False):
    """
    Construct inter-sample edges using scGALA between all patients and the centroid patient.
    Save as a dict: {patient: edge_index (2, N_edges)}.
    """
    if os.path.exists(save_path) and not force_recompute:
        with open(save_path, 'rb') as f:
            edge_dict = pickle.load(f)
        print(f"Loaded inter-sample edges from {save_path}")
        return edge_dict

    from .main import get_alignments
    patients = adata.obs[patient_key].unique()
    if centroid_patient is None:
        centroid_patient = select_centroid_patient(adata, method=centroid_method, patient_key=patient_key)
    edge_dict = {}
    centroid_idx = adata.obs[patient_key] == centroid_patient
    adata_centroid = adata[centroid_idx].copy()
    for p in patients:
        if p == centroid_patient:
            continue
        idx = adata.obs[patient_key] == p
        adata_other = adata[idx].copy()
        align_matrix = get_alignments(
            adata1=adata_other, adata2=adata_centroid, k=k, min_value=0.9, lamb=0.3, devices=devices,
            get_matrix=True, scale=True,spatial=spatial
        )
        src, tgt = align_matrix.nonzero()
        src_global = np.where(idx)[0][src]
        tgt_global = np.where(centroid_idx)[0][tgt]
        edge_index = np.stack([src_global, tgt_global], axis=0)
        edge_dict[p] = edge_index
    with open(save_path, 'wb') as f:
        pickle.dump(edge_dict, f)
    print(f"Saved inter-sample edges to {save_path}")
    return edge_dict

def load_intersample_edges(save_path):
    with open(save_path, 'rb') as f:
        edge_dict = pickle.load(f)
    return edge_dict

class TwoStageDataModule(L.LightningDataModule):
    def __init__(self, adata_sn: AnnData, adata_st: AnnData, k=20, save=True, 
                 mnn1=None, mnn2=None, batch_size=1,
                 sn_inter_edges_path=None, st_inter_edges_path=None,
                 sn_centroid=None, st_centroid=None, devices=[0], force_recompute=False,
                 patient_key='patient', centroid_method='pca'):
        super().__init__()
        self.batch_size = batch_size
        sc.pp.pca(adata_sn)
        sc.pp.pca(adata_st)
        self.adata_sn = adata_sn
        self.adata_st = adata_st
        self.k = k
        self.devices = devices
        self.force_recompute = force_recompute

        # Prepare gene order as before
        start_time = time.time()
        reordered_adata_sn, adata_st_common, n_matching_genes = reorder_adata_genes(
            adata_sn, adata_st
        )
        end_time = time.time()
        print(f"Time taken to reorder adata genes: {end_time - start_time:.4f} seconds")
        print(f'{n_matching_genes} matching genes and {adata_sn.shape[1]-n_matching_genes} only in SN data')
        self.n_matching_genes = n_matching_genes

        # --- Inter-sample edges for SN ---
        if sn_inter_edges_path is None:
            sn_inter_edges_path = './sn_inter_edges.pkl'
        sn_inter_edges = construct_and_save_intersample_edges(
            reordered_adata_sn, sn_inter_edges_path, k=k, centroid_patient=sn_centroid, devices=devices,
            force_recompute=force_recompute, patient_key=patient_key, centroid_method=centroid_method,spatial=False
        )

        # --- Inter-sample edges for ST ---
        if st_inter_edges_path is None:
            st_inter_edges_path = './st_inter_edges.pkl'
        st_inter_edges = construct_and_save_intersample_edges(
            adata_st_common, st_inter_edges_path, k=k, centroid_patient=st_centroid, devices=devices,
            force_recompute=force_recompute, patient_key=patient_key, centroid_method=centroid_method, spatial=True
        )

        # --- Intra-sample edges ---
        sn_patients = reordered_adata_sn.obs[patient_key].unique()
        sn_edges = []
        for p in sn_patients:
            idx = reordered_adata_sn.obs[patient_key] == p
            X = reordered_adata_sn.obsm['X_pca'][idx]
            local_indices = np.where(idx)[0]
            knn = kneighbors_graph(X, k, mode='distance').tocoo()
            sn_edges.append(np.stack([local_indices[knn.row], local_indices[knn.col]], axis=0))
        sn_edges = np.concatenate(sn_edges, axis=1) if sn_edges else np.zeros((2,0), dtype=int)

        st_patients = adata_st_common.obs[patient_key].unique()
        st_edges = []
        st_spatial_edges = []
        for p in st_patients:
            idx = adata_st_common.obs[patient_key] == p
            X = adata_st_common.obsm['X_pca'][idx]
            local_indices = np.where(idx)[0]
            knn = kneighbors_graph(X, k, mode='distance').tocoo()
            st_edges.append(np.stack([local_indices[knn.row], local_indices[knn.col]], axis=0))
            if 'spatial' in adata_st_common.obsm:
                spatial_X = adata_st_common.obsm['spatial'][idx]
                spatial_knn = kneighbors_graph(spatial_X, int(k/2), mode='distance').tocoo()
                st_spatial_edges.append(np.stack([local_indices[spatial_knn.row], local_indices[spatial_knn.col]], axis=0))
        st_edges = np.concatenate(st_edges, axis=1) if st_edges else np.zeros((2,0), dtype=int)
        st_spatial_edges = np.concatenate(st_spatial_edges, axis=1) if st_spatial_edges else np.zeros((2,0), dtype=int)

        # --- Combine all edges ---
        # SN intra + SN inter
        sn_all_edges = sn_edges
        if sn_inter_edges is not None:
            for arr in sn_inter_edges.values():
                sn_all_edges = np.concatenate([sn_all_edges, arr], axis=1)
        # ST intra + ST inter + spatial
        st_all_edges = st_edges
        if st_inter_edges is not None:
            for arr in st_inter_edges.values():
                st_all_edges = np.concatenate([st_all_edges, arr], axis=1)
        if st_spatial_edges.shape[1] > 0:
            st_all_edges = np.concatenate([st_all_edges, st_spatial_edges], axis=1)

        # --- Merge SN and ST graphs ---
        bias = reordered_adata_sn.shape[0]
        # SN edges: indices as is
        # ST edges: offset by bias
        st_all_edges_offset = st_all_edges + bias
        # MNN edges (if provided)
        if mnn1 is not None and mnn2 is not None:
            mnn_edges = np.stack([np.array(mnn1), np.array(mnn2)+bias], axis=0)
        else:
            mnn_edges = np.zeros((2,0), dtype=int)
        # Final edge_index
        edge_index = np.concatenate([sn_all_edges, st_all_edges_offset, mnn_edges], axis=1)
        edge_index = to_undirected(torch.from_numpy(edge_index)).to(torch.int32)

        # Node features
        data1_x = reordered_adata_sn.X.toarray() if hasattr(reordered_adata_sn.X, 'toarray') else reordered_adata_sn.X
        data2_x = adata_st_common.X.toarray() if hasattr(adata_st_common.X, 'toarray') else adata_st_common.X
        x = np.concatenate([data1_x, data2_x], axis=0)
        x = torch.from_numpy(x).to(torch.float32)
        self.x = x
        self.edge_index = edge_index
        self.bias = bias
        self.num_nodes = x.shape[0]

        if save:
            with open('var_names_two_stage.txt', 'w') as f:
                f.write(f'{n_matching_genes}\n')
                for var_name in reordered_adata_sn.var_names:
                    f.write(f'{var_name}\n')

    def setup(self, stage=None):
        self.data = pyg_data.Data(x=self.x, edge_index=self.edge_index, bias=self.bias)

    def train_dataloader(self):
        return pyg_data.DataLoader([self.data], batch_size=self.batch_size, shuffle=False)

    def val_dataloader(self):
        return pyg_data.DataLoader([self.data], batch_size=self.batch_size, shuffle=False)