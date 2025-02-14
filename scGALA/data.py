# For Cell Alignment
from anndata import AnnData
from torch.utils.data import Dataset,DataLoader
from .utils import get_graph
import lightning as L

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
        return DataLoader([self.data],batch_size=1)

    def val_dataloader(self):
        return DataLoader([self.data],batch_size=1)
    
    def predict_dataloader(self):
        return DataLoader([self.data],batch_size=1)

    def teardown(self, stage):
        # Clean up state after the trainer stops, delete files...
        if stage == 'predict':
            del self.data

    
def get_graph_spatial(data1:AnnData,data2:AnnData,mnn1,mnn2,k=20):
    # Get graph data
    edge_1 = kneighbors_graph(data1.obsm['X_pca'], k, mode='distance').tocoo()
    edge_2 = kneighbors_graph(data2.obsm['X_pca'], k, mode='distance').tocoo()
    
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
    data2_new[:,:data2.n_vars] = data2.X
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
    x = np.concatenate([data1.X, data2_new.X],axis=0) # get node features
    x = torch.from_numpy(x).to(torch.float32)
    num_nodes = data1.shape[0] + data2.shape[0]
    return x, edge_index, bias, num_nodes