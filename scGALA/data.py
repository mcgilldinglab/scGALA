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