# For Cell Alignment
from typing import Any, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import pytorch_lightning as pl
from scipy import sparse as sp
from torch_geometric.nn import (
    GCNConv,
    GATConv,
    GATv2Conv,
    ClusterGCNConv,
    AGNNConv,
    EGConv,
    SAGEConv,
    InnerProductDecoder,
    Sequential
)
from torch_geometric.utils import (
    negative_sampling,
    remove_self_loops,
    add_self_loops
)
from torch.nn import Linear, ReLU, BatchNorm1d, Dropout
from torch import Tensor
import anndata as ad
import numpy as np
import gc
from .utils import TypedEdgeRemoving,cross_dist,CosineLoss

# Constants
EPS = 1e-15
MAX_LOGSTD = 10

# class VariationalGCNEncoder(torch.nn.Module):
#     def __init__(self, in_channels=-1, out_channels=256, dropout = 0.3):
#         super(VariationalGCNEncoder, self).__init__()
#         self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True) # cached only for transductive learning
#         self.conv_mu = GCNConv(2 * out_channels, out_channels, cached=True)
#         self.conv_logstd = GCNConv(2 * out_channels, out_channels, cached=True)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index)
#         x = F.leaky_relu(self.dropout(x))
#         return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

# Replace the existing augmentor with our custom type-specific augmentor
# aug_feature_masking = A.FeatureMasking(pf=0.1)
aug_edge_removing = TypedEdgeRemoving( inter_pe=0.5)  # Default values

# class VGAE_gcl(L.LightningModule):
#     def __init__(self,in_channels:int = -1,out_channels:int = 256,dropout:float=0.3,lr:float=3e-4,use_scheduler:bool = True,optimizer:Literal['adam','sgd'] = 'adam') -> None:
#         super().__init__()
#         # self.x, self.edge_index, self.edge_weight, self.data = get_graph(data1,data2,k)
#         self.lr = lr
#         self.model = VGAE(VariationalGCNEncoder(in_channels=in_channels,out_channels=out_channels, dropout=dropout)) 
#         self.use_scheduler = use_scheduler
#         self.optimizer = optimizer
#     def training_step(self, batch, batch_idx):
#         x, edge_index, bias, num_nodes = batch
#         x, edge_index, bias, num_nodes = x[0], edge_index[0], bias[0], num_nodes[0]
#         # pdb.set_trace()
#         x_new, edge_index_new,_ = aug(x, edge_index)
#         z = self.model.encode(x_new,edge_index_new)
#         vae_loss = self.model.recon_loss(z, edge_index) 
#         vae_loss = vae_loss + (1 / num_nodes) * self.model.kl_loss()  # new line
#         self.log_dict({'train_loss':float(vae_loss)},prog_bar=True)
#         return vae_loss
#     def configure_optimizers(self):
#         if self.optimizer == 'adam':
#             optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
#         if self.optimizer == 'sgd':
#             optimizer = torch.optim.SGD(self.parameters(), lr=self.lr,momentum=0.9,nesterov=True)
#         if self.use_scheduler:
#             return {
#             "optimizer": optimizer,
#             "lr_scheduler": {
#                 "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.2,patience=2,threshold=5e-2,    threshold_mode='rel',verbose=True),
#                 "monitor": 'ave_align',
#                 "frequency": 1
#                 # If "monitor" references validation metrics, then "frequency" should be set to a
#                 # multiple of "trainer.check_val_every_n_epoch".
#             },
#             }
#         else:
#             return optimizer
#     def lr_scheduler_step(self, scheduler, metric) -> None:
#         if metric is None:
#             scheduler.step()
#         else:
#             scheduler.step(metric)
#     def validation_step(self, batch, batch_idx):
#         x, edge_index, bias, num_nodes = batch
#         x, edge_index, bias, num_nodes = x[0], edge_index[0], bias[0], num_nodes[0]
#         z = self.model.encode(x,edge_index)
#         neg_edge_index = negative_sampling(edge_index, z.size(0))
#         auc,ap = self.model.test(z, edge_index, neg_edge_index)
#         likelyhood = torch.cat([(z[i]*z).sum(dim=1).unsqueeze(0) for i in range(bias)],dim=0)[:bias,bias:].sigmoid()
#         self.log_dict({'auc':auc,'ap':ap,'ave_align':likelyhood[likelyhood>0.9].shape[0]/likelyhood.shape[0]},prog_bar=True)
#     # def forward(self,x,edge_index) -> Any:
#     #     return self.model.encode(x,edge_index)
#     def predict_step(self,batch, batch_idx) -> Any:
#         x, edge_index, bias, num_nodes = batch
#         x, edge_index, bias, num_nodes = x[0], edge_index[0], bias[0], num_nodes[0]
#         z = self.model.encode(x,edge_index)
#         return z

class GAT_Encoder(nn.Module):
    def __init__(self, num_heads, in_channels, latent_dim,hidden_dims=[64,128], dropout=0.4):
        super(GAT_Encoder, self).__init__()
        # initialize parameter
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        # initialize GAT layer
        self.hidden_layer_1 = GATConv(
            in_channels=in_channels, out_channels=hidden_dims[0],
            heads=self.num_heads[0],
            dropout=dropout,
            concat=True)
        in_dim2 = hidden_dims[0] * self.num_heads[0]# + in_channels * 2
        # in_dim2 = hidden_dims[0] * self.num_heads['first']

        self.hidden_layer_2 = GATConv(
            in_channels=in_dim2, out_channels=hidden_dims[1],
            heads=self.num_heads[1],
            dropout=dropout,
            concat=True)

        in_dim_final = hidden_dims[-1] * self.num_heads[1] #+ in_channels
        # in_dim_final = hidden_dims[-1] * self.num_heads['second']

        self.out_mean_layer = GATConv(in_channels=in_dim_final, out_channels=self.latent_dim,
                                      heads=self.num_heads[2], concat=False, dropout=0.4)
        self.out_logstd_layer = GATConv(in_channels=in_dim_final, out_channels=self.latent_dim,
                                        heads=self.num_heads[3], concat=False, dropout=0.4)

    def forward(self, x, edge_index):
        out = self.hidden_layer_1(x, edge_index)
        out = F.relu(out)
        # # add Gaussian noise being the same shape as x and concat
        # out = torch.cat([x, torch.randn_like(x), out], dim=1)
        out = self.hidden_layer_2(out, edge_index)
        out = F.relu(out)
        out = F.dropout(out, p=0.4, training=self.training)
        last_out = out
        # # concat x with last_out
        # last_out = torch.cat([x, last_out], dim=1)
        z_mean = self.out_mean_layer(last_out, edge_index)
        z_logstd = self.out_logstd_layer(last_out, edge_index)

        return z_mean, z_logstd


class GAT_Encoder_one_hidden(nn.Module):
    def __init__(self, num_heads, in_channels, latent_dim,hidden_dim=128, dropout=0.4):
        super().__init__()
        # initialize parameter
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        # initialize GAT layer
        self.hidden_layer = GATConv(
            in_channels=in_channels, out_channels=hidden_dim,
            heads=self.num_heads[1],
            dropout=dropout,
            concat=True)

        in_dim_final = hidden_dim * self.num_heads[1] #+ in_channels
        # in_dim_final = hidden_dims[-1] * self.num_heads['second']

        self.out_mean_layer = GATConv(in_channels=in_dim_final, out_channels=self.latent_dim,
                                      heads=self.num_heads[2], concat=False, dropout=dropout)
        self.out_logstd_layer = GATConv(in_channels=in_dim_final, out_channels=self.latent_dim,
                                        heads=self.num_heads[3], concat=False, dropout=dropout)

    def forward(self, x, edge_index):
        out = self.hidden_layer(x, edge_index)
        out = F.relu(out)
        # # add Gaussian noise being the same shape as x and concat
        # out = torch.cat([x, torch.randn_like(x), out], dim=1)
        last_out = out
        # # concat x with last_out
        # last_out = torch.cat([x, last_out], dim=1)
        z_mean = self.out_mean_layer(last_out, edge_index)
        z_logstd = self.out_logstd_layer(last_out, edge_index)

        return z_mean, z_logstd

class GAT_Encoder_no_hidden(nn.Module):
    def __init__(self, num_heads, in_channels, latent_dim,hidden_dim=128, dropout=0.4):
        super().__init__()
        # initialize parameter
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.num_heads = num_heads

        self.out_mean_layer = GATConv(in_channels=in_channels, out_channels=self.latent_dim,
                                      heads=self.num_heads[2], concat=False, dropout=dropout)
        self.out_logstd_layer = GATConv(in_channels=in_channels, out_channels=self.latent_dim,
                                        heads=self.num_heads[3], concat=False, dropout=dropout)

    def forward(self, x, edge_index):
        last_out = x
        # # concat x with last_out
        # last_out = torch.cat([x, last_out], dim=1)
        z_mean = self.out_mean_layer(last_out, edge_index)
        z_logstd = self.out_logstd_layer(last_out, edge_index)

        return z_mean, z_logstd


class GATv2_Encoder(nn.Module):
    def __init__(self, num_heads, in_channels, latent_dim, hidden_dims=[64,128], dropout=0.4):
        super(GATv2_Encoder, self).__init__()
        # initialize parameter
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        # initialize GATv2 layers
        self.hidden_layer_1 = GATv2Conv(
            in_channels=in_channels, out_channels=hidden_dims[0],
            heads=self.num_heads[0],
            dropout=dropout,
            concat=True)
        in_dim2 = hidden_dims[0] * self.num_heads[0]

        self.hidden_layer_2 = GATv2Conv(
            in_channels=in_dim2, out_channels=hidden_dims[1],
            heads=self.num_heads[1],
            dropout=dropout,
            concat=True)

        in_dim_final = hidden_dims[-1] * self.num_heads[1]

        self.out_mean_layer = GATv2Conv(in_channels=in_dim_final, out_channels=self.latent_dim,
                                        heads=self.num_heads[2], concat=False, dropout=0.4)
        self.out_logstd_layer = GATv2Conv(in_channels=in_dim_final, out_channels=self.latent_dim,
                                          heads=self.num_heads[3], concat=False, dropout=0.4)

    def forward(self, x, edge_index):
        out = self.hidden_layer_1(x, edge_index)
        out = F.relu(out)
        out = self.hidden_layer_2(out, edge_index)
        out = F.relu(out)
        out = F.dropout(out, p=0.4, training=self.training)
        last_out = out
        z_mean = self.out_mean_layer(last_out, edge_index)
        z_logstd = self.out_logstd_layer(last_out, edge_index)

        return z_mean, z_logstd


class SAGE_Encoder(nn.Module):
    def __init__(self, in_channels, latent_dim, hidden_dims=[64,128], dropout=0.4):
        super(SAGE_Encoder, self).__init__()
        # initialize parameter
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        # initialize SAGE layers
        self.hidden_layer_1 = SAGEConv(in_channels=in_channels, out_channels=hidden_dims[0])
        self.hidden_layer_2 = SAGEConv(in_channels=hidden_dims[0], out_channels=hidden_dims[1])
        self.out_mean_layer = SAGEConv(in_channels=hidden_dims[1], out_channels=self.latent_dim)
        self.out_logstd_layer = SAGEConv(in_channels=hidden_dims[1], out_channels=self.latent_dim)
        self.dropout = Dropout(dropout)

    def forward(self, x, edge_index):
        out = self.hidden_layer_1(x, edge_index)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.hidden_layer_2(out, edge_index)
        out = F.relu(out)
        out = self.dropout(out)
        last_out = out
        z_mean = self.out_mean_layer(last_out, edge_index)
        z_logstd = self.out_logstd_layer(last_out, edge_index)

        return z_mean, z_logstd


class ClusterGCN_Encoder(nn.Module):
    def __init__(self, in_channels, latent_dim, hidden_dims=[64,128], dropout=0.4):
        super(ClusterGCN_Encoder, self).__init__()
        # initialize parameter
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        # initialize ClusterGCN layers
        self.hidden_layer_1 = ClusterGCNConv(in_channels=in_channels, out_channels=hidden_dims[0])
        self.hidden_layer_2 = ClusterGCNConv(in_channels=hidden_dims[0], out_channels=hidden_dims[1])
        self.out_mean_layer = ClusterGCNConv(in_channels=hidden_dims[1], out_channels=self.latent_dim)
        self.out_logstd_layer = ClusterGCNConv(in_channels=hidden_dims[1], out_channels=self.latent_dim)
        self.dropout = Dropout(dropout)

    def forward(self, x, edge_index):
        out = self.hidden_layer_1(x, edge_index)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.hidden_layer_2(out, edge_index)
        out = F.relu(out)
        out = self.dropout(out)
        last_out = out
        z_mean = self.out_mean_layer(last_out, edge_index)
        z_logstd = self.out_logstd_layer(last_out, edge_index)

        return z_mean, z_logstd

    
class MSVGAE_gcl(L.LightningModule):
    def __init__(self,in_channels:int = -1,out_channels:list = [16,32,64],out_dim=64,dropout:float=0.3,lr:float=3e-4, masking_ratio=0.3,use_scheduler:bool = True,optimizer:Literal['adam','sgd'] = 'adam',version = 'normal',inter_edge_mask_weight:float = 0.5,layer_type:Literal['GAT', 'GATv2', 'SAGE', 'ClusterGCN'] = 'GAT') -> None:
        super().__init__()
        # self.x, self.edge_index, self.edge_weight, self.data = get_graph(data1,data2,k)
        self.lr = lr
        
        # Create encoder list based on layer_type
        encoders = []
        if layer_type == 'GAT':
            if version == 'normal':
                encoders = [GAT_Encoder(num_heads=[4,4,4,4],in_channels=in_channels,latent_dim=out_channels[i], dropout=dropout) for i in range(len(out_channels))]
            elif version == 'simple':
                encoders = [GAT_Encoder(num_heads=[1,1,1,1],in_channels=in_channels,latent_dim=out_channels[i], dropout=dropout) for i in range(len(out_channels))]
            elif version == 'naive':
                encoders = [GAT_Encoder_no_hidden(num_heads=[1,1,1,1],in_channels=in_channels,latent_dim=out_channels[i], dropout=dropout) for i in range(len(out_channels))]
        elif layer_type == 'GATv2':
            if version == 'normal':
                encoders = [GATv2_Encoder(num_heads=[4,4,4,4],in_channels=in_channels,latent_dim=out_channels[i], dropout=dropout) for i in range(len(out_channels))]
            elif version == 'simple':
                encoders = [GATv2_Encoder(num_heads=[1,1,1,1],in_channels=in_channels,latent_dim=out_channels[i], dropout=dropout) for i in range(len(out_channels))]
            elif version == 'naive':
                raise ValueError("GATv2 doesn't support 'naive' version without hidden layer")
        elif layer_type == 'SAGE':
            if version == 'normal':
                encoders = [SAGE_Encoder(in_channels=in_channels,latent_dim=out_channels[i], hidden_dims=[64,128], dropout=dropout) for i in range(len(out_channels))]
            elif version == 'simple':
                encoders = [SAGE_Encoder(in_channels=in_channels,latent_dim=out_channels[i], hidden_dims=[32,64], dropout=dropout) for i in range(len(out_channels))]
            elif version == 'naive':
                raise ValueError("SAGE doesn't support 'naive' version without hidden layer")
        elif layer_type == 'ClusterGCN':
            if version == 'normal':
                encoders = [ClusterGCN_Encoder(in_channels=in_channels,latent_dim=out_channels[i], hidden_dims=[64,128], dropout=dropout) for i in range(len(out_channels))]
            elif version == 'simple':
                encoders = [ClusterGCN_Encoder(in_channels=in_channels,latent_dim=out_channels[i], hidden_dims=[32,64], dropout=dropout) for i in range(len(out_channels))]
            elif version == 'naive':
                raise ValueError("ClusterGCN doesn't support 'naive' version without hidden layer")
        else:
            raise ValueError(f"Unsupported layer_type: {layer_type}. Choose from 'GAT', 'GATv2', 'SAGE', 'ClusterGCN'")
        
        self.model = MSVGAE(nn.ModuleList(encoders), out_dim=out_dim)
        self.use_scheduler = use_scheduler
        self.optimizer = optimizer
        # self.aug = A.RandomChoice([
        #             A.FeatureMasking(pf=0.1),
        #             A.EdgeRemoving(pe=masking_ratio)],
        #             num_choices=1)  ##A.NodeDropping(pn=0.1),
        # Setup the edge augmentor with the specified weight for inter-dataset edges
        self.edge_augmentor = TypedEdgeRemoving( inter_pe=inter_edge_mask_weight, total_pe=masking_ratio)
    def training_step(self, batch, batch_idx):
        if len(batch) == 5:  # Non-spatial case
            x, edge_index, bias, num_nodes, edge_type = batch
            x, edge_index, bias, num_nodes, edge_type = x[0], edge_index[0], bias[0], num_nodes[0], edge_type[0]
            
            # Apply feature masking
            mask = torch.FloatTensor(x.shape[0], x.shape[1]).uniform_() > 0.1
            x_new = x * mask.to(x.device)
            
            # Apply edge masking with different weights for different edge types
            _, edge_index_new, edge_type_new = self.edge_augmentor(x, edge_index, edge_type)
            
            z = self.model.encode(x_new, edge_index_new)
        else: raise ValueError(f"The input data format is incorrect. It should contain 5 elements including edge types. Now it has {len(batch)} elements.")
        
        vae_loss = self.model.recon_loss(z, edge_index) 
        reconstructed_features = self.model.liner_decoder(z)
        decoder_loss = torch.nn.functional.mse_loss(reconstructed_features, x) * 10
        vae_loss = vae_loss + (1 / num_nodes) * self.model.kl_loss() + decoder_loss # new line
        self.log_dict({'train_loss':float(vae_loss),'decoder_loss':float(decoder_loss)},prog_bar=True)
        return vae_loss
    def configure_optimizers(self):
        if self.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        if self.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr,momentum=0.9,nesterov=True)
        if self.use_scheduler:
            return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=0.2,patience=2,threshold=3e-3,    threshold_mode='rel',verbose=True),
                "monitor": 'ap',
                "frequency": 1
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
            }
        else:
            return optimizer
    def lr_scheduler_step(self, scheduler, metric) -> None:
        if metric is None:
            scheduler.step()
        else:
            scheduler.step(metric)
    def validation_step(self, batch, batch_idx):
        if len(batch) == 5:  # Non-spatial with edge_type
            x, edge_index, bias, num_nodes, edge_type = batch
            x, edge_index, bias, num_nodes = x[0], edge_index[0], bias[0], num_nodes[0]
        else:
            x, edge_index, bias, num_nodes = batch
            x, edge_index, bias, num_nodes = x[0], edge_index[0], bias[0], num_nodes[0]
        z = self.model.encode(x, edge_index)
        neg_edge_index = negative_sampling(edge_index, z.size(0))
        auc, ap = self.model.test(z, edge_index, neg_edge_index)
        likelyhood = torch.matmul(z[:bias], z[bias:].T).sigmoid()
        self.log_dict({'auc': auc, 'ap': ap, 'ave_align': likelyhood[likelyhood > 0.9].shape[0] / likelyhood.shape[0]}, prog_bar=True)
    # def forward(self,x,edge_index) -> Any:
    #     return self.model.encode(x,edge_index)
    def predict_step(self, batch, batch_idx) -> Any:
        if len(batch) == 5:  # Non-spatial with edge_type
            x, edge_index, bias, num_nodes, edge_type = batch
            x, edge_index, bias, num_nodes = x[0], edge_index[0], bias[0], num_nodes[0]
        else:
            x, edge_index, bias, num_nodes = batch
            x, edge_index, bias, num_nodes = x[0], edge_index[0], bias[0], num_nodes[0]
        z = self.model.encode(x, edge_index)
        return z

class MSVGAE_gcl_spatialGW(L.LightningModule):
    def __init__(self,in_channels:int = -1,out_channels:list = [16,32,64],out_dim=64,dropout:float=0.3,lr:float=3e-4, masking_ratio=0.3,use_scheduler:bool = True,optimizer:Literal['adam','sgd'] = 'adam',version:Literal['normal','simple','naive'] = 'normal',inter_edge_mask_weight:float = 0.5,layer_type:Literal['GAT', 'GATv2', 'SAGE', 'ClusterGCN'] = 'GAT') -> None:
        super().__init__()
        # self.x, self.edge_index, self.edge_weight, self.data = get_graph(data1,data2,k)
        self.lr = lr
        
        # Create encoder list based on layer_type
        encoders = []
        if layer_type == 'GAT':
            if version == 'normal':
                encoders = [GAT_Encoder(num_heads=[4,4,4,4],in_channels=in_channels,latent_dim=out_channels[i], dropout=dropout) for i in range(len(out_channels))]
            elif version == 'simple':
                encoders = [GAT_Encoder(num_heads=[1,1,1,1],in_channels=in_channels,latent_dim=out_channels[i], dropout=dropout) for i in range(len(out_channels))]
            elif version == 'naive':
                encoders = [GAT_Encoder_no_hidden(num_heads=[1,1,1,1],in_channels=in_channels,latent_dim=out_channels[i], dropout=dropout) for i in range(len(out_channels))]
        elif layer_type == 'GATv2':
            if version == 'normal':
                encoders = [GATv2_Encoder(num_heads=[4,4,4,4],in_channels=in_channels,latent_dim=out_channels[i], dropout=dropout) for i in range(len(out_channels))]
            elif version == 'simple':
                encoders = [GATv2_Encoder(num_heads=[1,1,1,1],in_channels=in_channels,latent_dim=out_channels[i], dropout=dropout) for i in range(len(out_channels))]
            elif version == 'naive':
                raise ValueError("GATv2 doesn't support 'naive' version without hidden layer")
        elif layer_type == 'SAGE':
            if version == 'normal':
                encoders = [SAGE_Encoder(in_channels=in_channels,latent_dim=out_channels[i], hidden_dims=[64,128], dropout=dropout) for i in range(len(out_channels))]
            elif version == 'simple':
                encoders = [SAGE_Encoder(in_channels=in_channels,latent_dim=out_channels[i], hidden_dims=[32,64], dropout=dropout) for i in range(len(out_channels))]
            elif version == 'naive':
                raise ValueError("SAGE doesn't support 'naive' version without hidden layer")
        elif layer_type == 'ClusterGCN':
            if version == 'normal':
                encoders = [ClusterGCN_Encoder(in_channels=in_channels,latent_dim=out_channels[i], hidden_dims=[64,128], dropout=dropout) for i in range(len(out_channels))]
            elif version == 'simple':
                encoders = [ClusterGCN_Encoder(in_channels=in_channels,latent_dim=out_channels[i], hidden_dims=[32,64], dropout=dropout) for i in range(len(out_channels))]
            elif version == 'naive':
                raise ValueError("ClusterGCN doesn't support 'naive' version without hidden layer")
        else:
            raise ValueError(f"Unsupported layer_type: {layer_type}. Choose from 'GAT', 'GATv2', 'SAGE', 'ClusterGCN'")
        
        self.model = MSVGAE(nn.ModuleList(encoders), out_dim=out_dim)
        self.use_scheduler = use_scheduler
        self.optimizer = optimizer    
        # Setup the edge augmentor with the specified weight for inter-dataset edges
        self.edge_augmentor = TypedEdgeRemoving( inter_pe=inter_edge_mask_weight, total_pe=masking_ratio)
        # try:
        #     import ot
        #     self.ot = ot
        # except ImportError:
        #     raise ImportError('\nplease install pot:\n\tpip install POT')
        # # Define the triplet margin loss criterion
        self.criterion = nn.TripletMarginLoss(margin=1.0)
    def training_step(self, batch, batch_idx):
        if len(batch) == 9:  # Updated format with edge types
            x, edge_index, bias, num_nodes, C1, C2, spatial_edge_index, edge_type, spatial_edge_type = batch
            x, edge_index, bias, num_nodes, C1, C2, spatial_edge_index, edge_type, spatial_edge_type = (
                x[0], edge_index[0], bias[0], num_nodes[0], C1[0], C2[0], 
                spatial_edge_index[0], edge_type[0], spatial_edge_type[0]
            )
            
            # Apply feature masking
            mask = torch.FloatTensor(x.shape[0], x.shape[1]).uniform_() > 0.1
            x_new = x * mask.to(x.device)
            
            # Apply edge masking with different weights for different edge types
            _, edge_index_new, edge_type_new = self.edge_augmentor(x, edge_index, edge_type)
            
            z = self.model.encode(x_new, edge_index_new)
            
        else: raise ValueError(f"The input data format is incorrect. It should contain 9 elements including edge types. Now it has {len(batch)} elements.")
        
        vae_loss = self.model.recon_loss(z, edge_index) 
        spatial_triplet_loss = self.compute_triplet_loss(z, spatial_edge_index)*6
        total_loss = vae_loss + (1 / num_nodes) * self.model.kl_loss() + spatial_triplet_loss
        
        self.log_dict({
            'total_loss': float(total_loss),
            'vae_loss': float(vae_loss),
            'spatial_triplet_loss': float(spatial_triplet_loss)
        }, prog_bar=True)
        
        return total_loss
    def configure_optimizers(self):
        if self.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        if self.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr,momentum=0.9,nesterov=True)
        if self.use_scheduler:
            return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=0.2,patience=2,threshold=3e-3,    threshold_mode='rel',verbose=True),
                "monitor": 'ap',
                "frequency": 1
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
            }
        else:
            return optimizer
    def lr_scheduler_step(self, scheduler, metric) -> None:
        if metric is None:
            scheduler.step()
        else:
            scheduler.step(metric)
    def validation_step(self, batch, batch_idx):
        if len(batch) == 9:  # With edge_type and spatial_edge_type
            x, edge_index, bias, num_nodes, C1, C2, spatial_edge_index, edge_type, spatial_edge_type = batch
            x, edge_index, bias, num_nodes = x[0], edge_index[0], bias[0], num_nodes[0]
        else:
            x, edge_index, bias, num_nodes, C1, C2, spatial_edge_index = batch
            x, edge_index, bias, num_nodes = x[0], edge_index[0], bias[0], num_nodes[0]
        z = self.model.encode(x, edge_index)
        neg_edge_index = negative_sampling(edge_index, z.size(0))
        auc, ap = self.model.test(z, edge_index, neg_edge_index)
        likelyhood = torch.matmul(z[:bias], z[bias:].T).sigmoid()
        self.log_dict({'auc': auc, 'ap': ap, 'ave_align': likelyhood[likelyhood > 0.9].shape[0] / likelyhood.shape[0]}, prog_bar=True)
    # def forward(self,x,edge_index) -> Any:
    #     return self.model.encode(x,edge_index)
    def predict_step(self, batch, batch_idx) -> Any:
        if len(batch) == 9:  # With edge_type and spatial_edge_type
            x, edge_index, bias, num_nodes, C1, C2, spatial_edge_index, edge_type, spatial_edge_type = batch
            x, edge_index, bias, num_nodes = x[0], edge_index[0], bias[0], num_nodes[0]
        else:
            x, edge_index, bias, num_nodes, C1, C2, spatial_edge_index = batch
            x, edge_index, bias, num_nodes = x[0], edge_index[0], bias[0], num_nodes[0]
        z = self.model.encode(x, edge_index)
        return z

    def compute_triplet_loss(self,embeddings, edge_index):
        """
        Compute triplet loss.
    
        Args:
            embeddings (torch.Tensor): Tensor of shape (N, D) representing node embeddings, where N is the number of nodes
                                       and D is the embedding dimension.
            edge_index (torch.Tensor): Tensor of shape (2, E) representing the COO-formatted edge index, where E is the number
                                       of edges.
            margin (float, optional): Margin for the triplet loss. Default is 1.0.
    
        Returns:
            torch.Tensor: Computed triplet loss.
    
        Note:
            This function assumes that the input edge index is provided in COO format, i.e., with two rows where the first row
            represents the source nodes and the second row represents the target nodes.
        """
        # Get source and target nodes from the edge index
        src_nodes = edge_index[0]
        tgt_nodes = edge_index[1]
    
        # Calculate all possible positive pairs
        pos_pairs = torch.stack([src_nodes, tgt_nodes], dim=1)
    
        # Randomly select a node as negative sample
        num_nodes = embeddings.size(0)
        neg_nodes = torch.randint(num_nodes, size=(edge_index.size(1),))

    
        # Calculate triplet loss
        loss = self.criterion(embeddings[pos_pairs[:, 0]], embeddings[pos_pairs[:, 1]], embeddings[neg_nodes])
    
        return loss

class MSVGAE(torch.nn.Module):
    def __init__(self, encoders, line_decoder_hid_dim=128,out_dim=64):
        super(MSVGAE, self).__init__()

        # # initialize parameter
        # self.mu_gat2 = self.logstd_gat2 = None
        # self.mu_gat1 = self.logstd_gat1 = None
        self.mus=[]
        self.logstds=[]
        # # encoder
        # self.encoder_gat1 = encoder_gat1
        # self.encoder_gat2 = encoder_gat2
        self.encoders = encoders
        self.num_encoders = len(encoders)
        encoded_dim = sum(self.encoders[i].latent_dim for i in range(self.num_encoders))
        # use inner product decoder by default
        self.decoder = InnerProductDecoder()
        # liner decoder
        self.liner_decoder = nn.Sequential(
            Linear(in_features=out_dim, out_features=line_decoder_hid_dim),
            BatchNorm1d(line_decoder_hid_dim),
            ReLU(),
            Dropout(0.4),
            Linear(in_features=line_decoder_hid_dim, out_features=self.encoders[-1].in_channels),
        )
        self.out_layer = Linear(in_features=encoded_dim, out_features=out_dim)

    def encode(self, *args, **kwargs):
        # """ encode """
        # # GAT encoder
        # self.mu_gat2, self.logstd_gat2 = self.encoder_gat2(*args, **kwargs)
        # # GCN encoder
        # self.mu_gat1, self.logstd_gat1 = self.encoder_gat1(*args, **kwargs)
        z=[]
        for encoder in self.encoders:
            mu,logstd = encoder(*args, **kwargs)
            self.mus.append(mu)
            self.logstds.append(logstd)
        # fix range
        for i in range(self.num_encoders):
            self.logstds[i] = self.logstds[i].clamp(max=MAX_LOGSTD)
        # reparameter
        for i in range(self.num_encoders):
            z.append(self.reparametrize(self.mus[i], self.logstds[i]))
        z = torch.concat(z, dim=1)
        z = self.out_layer(z)
        return z

    def reparametrize(self, mu, log_std):
        if self.training:
            return mu + torch.randn_like(log_std) * torch.exp(log_std)
        else:
            return mu

    def kl_loss(self):
        r"""Computes the KL loss, either for the passed arguments :obj:`mu`
        and :obj:`logstd`, or based on latent variables from last encoding.

        Args:
            mu (Tensor, optional): The latent space for :math:`\mu`. If set to
                :obj:`None`, uses the last computation of :math:`mu`.
                (default: :obj:`None`)
            logstd (Tensor, optional): The latent space for
                :math:`\log\sigma`.  If set to :obj:`None`, uses the last
                computation of :math:`\log\sigma^2`.(default: :obj:`None`)
        """
        loss_kl = 0.0
        for i in range(self.num_encoders):
            loss_kl += -0.5 * torch.mean(torch.sum(1 + 2 * self.logstds[i] - self.mus[i] ** 2 - self.logstds[i].exp()**2, dim=1))
        self.mus=[]
        self.logstds=[]
        return loss_kl / self.num_encoders

    def recon_loss(self, z, pos_edge_index, neg_edge_index=None):
        r"""Given latent variables :obj:`z`, computes the binary cross
        entropy loss for positive edges :obj:`pos_edge_index` and negative
        sampled edges.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to train against.
            neg_edge_index (LongTensor, optional): The negative edges to train
                against. If not given, uses negative sampling to calculate
                negative edges. (default: :obj:`None`)
        """

        self.decoded = self.decoder(z, pos_edge_index, sigmoid=True)
        pos_loss = -torch.log(self.decoded + EPS).mean()

        # Do not include self-loops in negative samples
        pos_edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 - self.decoder(z, neg_edge_index, sigmoid=True) + EPS).mean()

        return pos_loss + neg_loss
    def test(self, z: Tensor, pos_edge_index: Tensor,
             neg_edge_index: Tensor) :
        r"""Given latent variables :obj:`z`, positive edges
        :obj:`pos_edge_index` and negative edges :obj:`neg_edge_index`,
        computes area under the ROC curve (AUC) and average precision (AP)
        scores.

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (torch.Tensor): The positive edges to evaluate
                against.
            neg_edge_index (torch.Tensor): The negative edges to evaluate
                against.
        """
        from sklearn.metrics import average_precision_score, roc_auc_score

        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.decoder(z, pos_edge_index, sigmoid=True)
        neg_pred = self.decoder(z, neg_edge_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

        return roc_auc_score(y, pred), average_precision_score(y, pred)

# For Multiomics Generation

class GATMapper(pl.LightningModule):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0.2):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout)
        self.gat3 = GATConv(hidden_channels * heads, out_channels, heads=1, dropout=dropout)
        
    def forward(self, x, edge_index, edge_attr):
        x = F.elu(self.gat1(x, edge_index, edge_attr))
        x = F.elu(self.gat2(x, edge_index, edge_attr))
        x = F.relu(self.gat3(x, edge_index, edge_attr))
        return x
    
    def anchor_loss(self, pred, target, anchor_map):
        # Calculate loss only for anchor pairs
        pred_anchor = pred[anchor_map['atac_idx']]
        target_anchor = target[anchor_map['rna_idx']]
        weights = anchor_map['weights']
        
        # Weighted MSE loss
        loss = F.mse_loss(pred_anchor, target_anchor, reduction='none')
        loss = (loss.mean(dim=1) * weights).mean()
        return loss
    
    def training_step(self, batch, batch_idx):
        pred = self(batch.x, batch.edge_index, batch.edge_attr)
        loss = self.anchor_loss(pred, batch.y, batch.anchor_map)
        self.log('train_loss', loss,prog_bar=True,batch_size=1)
        return loss
    
    def validation_step(self, batch, batch_idx):
        pred = self(batch.x, batch.edge_index, batch.edge_attr)
        loss = self.anchor_loss(pred, batch.y, batch.anchor_map)
        self.log('val_loss', loss,prog_bar=True,batch_size=1)
        
    def predict_step(self, batch, batch_idx):
        pred = self(batch.x, batch.edge_index, batch.edge_attr)
        pred = pred.cpu().numpy()
        return pred
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

# For Spatial Imputation

class GNNImputer(L.LightningModule):
    def __init__(self, num_features,n_matching_genes, hidden_channels,heads=4,dropout=0.6,num_layers=3,learning_rate=1e-3,layer_type='GAT'):
        super(GNNImputer, self).__init__()
        if layer_type == 'GAT':
            conv_layer = GATConv
            GAT = True
        elif layer_type == 'GCN':
            conv_layer = GCNConv
            GAT = False
        elif layer_type == 'GATv2':
            conv_layer = GATv2Conv
            GAT = True
        elif layer_type == 'ClusterGCN':
            conv_layer = ClusterGCNConv
            GAT = False
        elif layer_type == 'AGNN':
            conv_layer = AGNNConv
            GAT = False
        elif layer_type == 'EGConv':
            conv_layer = EGConv
            GAT = False
        elif layer_type == 'SAGE':
            conv_layer = SAGEConv
            GAT = False
        else:
            raise ValueError(f"Invalid layer type: {layer_type}")
        
        self.convs_list = []
        if GAT:
            self.convs_list.append((conv_layer(num_features, hidden_channels, heads=heads, dropout=dropout),'x, edge_index -> x'))
            self.convs_list.append((torch.nn.ReLU(),'x -> x'))
            for _ in range(num_layers - 2):
                self.convs_list.append((conv_layer(hidden_channels*heads, hidden_channels, heads=heads, dropout=dropout),'x, edge_index -> x'))
                self.convs_list.append((torch.nn.ReLU(),'x -> x'))
            self.convs_list.append((conv_layer(hidden_channels*heads, num_features),'x, edge_index -> x'))
            self.convs_list.append((torch.nn.ReLU(),'x -> x'))
        elif layer_type == 'EGConv':
            assert hidden_channels % heads == 0, "hidden_channels must be divisible by heads for EGConv"
            self.convs_list.append((torch.nn.Dropout(dropout),'x -> x'))
            self.convs_list.append((conv_layer(num_features, hidden_channels,num_heads=heads,num_bases=heads,cached=True),'x, edge_index -> x'))
            self.convs_list.append((torch.nn.ReLU(),'x -> x'))
            for _ in range(num_layers - 2):
                self.convs_list.append((torch.nn.Dropout(dropout),'x -> x'))
                self.convs_list.append((conv_layer(hidden_channels, hidden_channels,num_heads=heads,num_bases=heads,cached=True),'x, edge_index -> x'))
                self.convs_list.append((torch.nn.ReLU(),'x -> x'))
            self.convs_list.append((torch.nn.Dropout(dropout),'x -> x'))
            self.convs_list.append((conv_layer(hidden_channels, num_features,num_heads=heads,num_bases=heads,cached=True),'x, edge_index -> x'))
            self.convs_list.append((torch.nn.ReLU(),'x -> x'))
        else:
            self.convs_list.append((torch.nn.Dropout(dropout),'x -> x'))
            self.convs_list.append((conv_layer(num_features, hidden_channels),'x, edge_index -> x'))
            self.convs_list.append((torch.nn.ReLU(),'x -> x'))
            for _ in range(num_layers - 2):
                self.convs_list.append((torch.nn.Dropout(dropout),'x -> x'))
                self.convs_list.append((conv_layer(hidden_channels, hidden_channels),'x, edge_index -> x'))
                self.convs_list.append((torch.nn.ReLU(),'x -> x'))
            self.convs_list.append((torch.nn.Dropout(dropout),'x -> x'))
            self.convs_list.append((conv_layer(hidden_channels, num_features),'x, edge_index -> x'))
            self.convs_list.append((torch.nn.ReLU(),'x -> x'))
        self.convs = Sequential('x, edge_index',self.convs_list)
        self.n_matching_genes = n_matching_genes
        self.learning_rate = learning_rate
    def forward(self, x, edge_index):
        x = self.convs(x, edge_index)
        return x
    
    def training_step(self, batch, batch_idx):
        x, edge_index,bias = batch.x, batch.edge_index,batch.bias
        x_hat = self(x, edge_index)
        loss_RNA = F.mse_loss(x_hat[:bias], x[:bias])
        loss_ST = F.mse_loss(x_hat[bias:,:self.n_matching_genes], x[bias:,:self.n_matching_genes])
        loss = loss_RNA + loss_ST
        self.log('train_loss', loss,batch_size=1,prog_bar=True)
        self.log('loss_RNA', loss_RNA,batch_size=1,prog_bar=True)
        self.log('loss_ST', loss_ST,batch_size=1,prog_bar=True)
        return loss
    
    def predict_step(self, batch, batch_idx):
        x, edge_index,bias = batch.x, batch.edge_index,batch.bias
        x_hat = self(x, edge_index)
        x_hat = x_hat.cpu().numpy()
        return x_hat,bias
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)
        return [optimizer], [lr_scheduler]

# For Improved Spatial Imputation with scGALA
class TwoStageGNNImputer_old(L.LightningModule):
    def __init__(self, num_features, n_matching_genes, hidden_channels=512, num_layers=3, 
                 layer_type='ClusterGCN', dropout=0.3, lr=1e-3, stage1_epochs=500, 
                 similarity_weight=0.5, alignment_lr=1e-4, alignment_devices=None,
                 alignment_update_freq_delta=50, stage2_patience=20, stage2_min_delta=1e-4,
                 stage1_patience=10, stage1_min_delta=1e-4, lam_genegraph=0.1,
                 discriminator_hidden=256, discriminator_lr=1e-3, adv_weight=0.05,
                 discriminator_steps=1, generator_steps=1):
        super().__init__()
        self.save_hyperparameters()
        
        # Main imputation model
        self.imputer = GNNImputer(num_features=num_features, n_matching_genes=n_matching_genes, hidden_channels=hidden_channels, 
                                  num_layers=num_layers, layer_type=layer_type, dropout=dropout, learning_rate=lr)
        
        # Two-stage training parameters
        self.stage1_epochs = stage1_epochs
        self.similarity_weight = similarity_weight
        self.alignment_lr = alignment_lr
        self.alignment_devices = alignment_devices if alignment_devices is not None else [0]
        self.current_epoch_stage = 0

        # Early stopping parameters for stage 1
        self.stage1_patience = stage1_patience
        self.stage1_min_delta = stage1_min_delta
        self.stage1_best_loss = float('inf')
        self.stage1_wait = 0
        self.stage1_stopped = False

        # Early stopping parameters for stage 2
        self.stage2_patience = stage2_patience
        self.stage2_min_delta = stage2_min_delta
        self.stage2_best_loss = float('inf')
        self.stage2_wait = 0
        self.stage2_stopped = False
        
        # For storing intermediate results
        self.stage1_complete = False
        self.sn_indices = None
        self.st_indices = None
        self.alignment_update_freq = 1
        self.alignment_update_freq_delta = alignment_update_freq_delta
        self.sn_sn_similarity = None
        self.lam_genegraph = lam_genegraph
        self.genegraph_loss = CosineLoss()  # Initialize the gene graph loss
        self.sn_genegraph = None  # Cache for SN_genegraph
        self.automatic_optimization = False
        disc_in_features = max(1, int(self.hparams.num_features))
        hidden_mid = max(1, discriminator_hidden // 2)
        self.discriminator = nn.Sequential(
            nn.Linear(disc_in_features, discriminator_hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(discriminator_hidden, hidden_mid),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_mid, 1)
        )
        self.adv_loss_fn = nn.BCEWithLogitsLoss()
        self.generator_steps = generator_steps
        self.discriminator_steps = discriminator_steps

    def setup_indices(self, sn_size, st_size):
        """Setup indices for SN and ST data"""
        sn_size = int(sn_size)
        st_size = int(st_size)
        if sn_size <= 0 or st_size <= 0:
            raise ValueError("sn_size and st_size must be positive integers.")
        self.sn_indices = torch.arange(sn_size)
        self.st_indices = torch.arange(sn_size, sn_size + st_size)
        
    def forward(self, x, edge_index):
        return self.imputer(x, edge_index)
    
    def compute_alignment_matrices(self, sn_data, st_data):
        """Compute alignment matrices using scGALA's get_alignments function"""
        from .main import get_alignments
        
        # Create temporary AnnData objects
        sn_adata = ad.AnnData(X=sn_data.detach().cpu().numpy())
        st_adata = ad.AnnData(X=st_data.detach().cpu().numpy())
        
        # Get alignment matrix using scGALA
        alignment_matrix = get_alignments(
            adata1=sn_adata, 
            adata2=st_adata,
            k=20,
            min_value=0.8, 
            lr=self.alignment_lr,
            max_epochs=10,  # Fewer epochs for efficiency
            get_edge_probs=True,
            scale=True,
            devices=self.alignment_devices,
            default_root_dir='./logs/scgala_alignment',
        )

        # Add explicit GPU memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Clean up temporary AnnData objects
        del sn_adata, st_adata
        gc.collect()
        
        print('Alignment matrix computed at epoch:', self.current_epoch_stage)
        return torch.tensor(alignment_matrix, device=self.device, dtype=torch.float32)
    
    def compute_similarity_matrices(self, data, k=20, sparse=True):
        """Compute sparse pairwise similarity matrices using cosine similarity with K-NN, preserving gradients."""
        # Normalize data
        data_norm = F.normalize(data, p=2, dim=1)
        
        # Compute full cosine similarity for finding K-NN
        full_similarity = torch.mm(data_norm, data_norm.t())
        
        if sparse:
            # Find top-k similarities for each node (including self)
            topk_values, topk_indices = torch.topk(full_similarity, k=k+1, dim=1, largest=True)

            # Create a mask for the K-NN values
            n_nodes = data.size(0)
            mask = torch.zeros_like(full_similarity, dtype=torch.bool)
            mask.scatter_(1, topk_indices, True)

            # Use the mask to keep only K-NN similarities, set others to zero (preserves gradient)
            similarity = torch.where(mask, full_similarity, torch.zeros_like(full_similarity))
        else:
            # Use full similarity matrix directly
            similarity = full_similarity
        # Make symmetric by taking max(sim[i,j], sim[j,i])
        similarity = torch.max(similarity, similarity.t())
        
        return similarity
    
    def compute_similarity_loss(self, sn_data, st_data, alignment_matrix):
        """Compute similarity preservation loss"""
        # Get alignment probabilities between SN and ST
        if not hasattr(self, 'st_sn_alignment_norm'):
            sn_st_alignment = alignment_matrix
            st_sn_alignment = alignment_matrix.t()
            # Row-normalize alignment matrices (each row sums to 1)
            self.st_sn_alignment_norm = F.normalize(st_sn_alignment, p=1, dim=1).detach()
            self.sn_st_alignment_norm = F.normalize(sn_st_alignment, p=1, dim=1).detach()
            
        if (self.current_epoch - self.stage1_epochs) % self.alignment_update_freq == 0:
            # Clear previous alignment matrix to free memory
            if hasattr(self, 'sn_st_alignment'):
                del self.sn_st_alignment
                torch.cuda.empty_cache()
            self.sn_st_alignment = self.compute_alignment_matrices(sn_data, st_data)
            self.alignment_update_freq += self.alignment_update_freq_delta
            self.alignment_update_freq_delta += self.alignment_update_freq_delta
            # This can make similarity loss suddenly large if alignment changes significantly, so need to reset early stopping
            self.stage2_best_loss = float('inf')
            st_sn_alignment = self.sn_st_alignment.t()  # Transpose to get ST-SN
            # Row-normalize alignment matrices (each row sums to 1)
            self.st_sn_alignment_norm = F.normalize(st_sn_alignment, p=1, dim=1).detach()
            self.sn_st_alignment_norm = F.normalize(self.sn_st_alignment, p=1, dim=1).detach()
            print('Alignment matrices updated at epoch:', self.current_epoch)

        # Compute similarity matrices
        if self.sn_sn_similarity is None:
            self.sn_sn_similarity = self.compute_similarity_matrices(sn_data,sparse=True,k=20)
            self.sn_sn_similarity = self.sn_sn_similarity.detach()  # Detach to prevent gradients
            actual_flat_sn = self.sn_sn_similarity.flatten()
            self.actual_norm_sn = F.normalize(actual_flat_sn.unsqueeze(0), p=2, dim=1)
            
        st_st_similarity = self.compute_similarity_matrices(st_data, sparse=False,k=20)
        
        # Expected ST-ST similarity based on SN-SN similarity and SN-ST alignment
        # expected_st_st = ST-SN @ SN-SN @ SN-ST
        
        expected_st_st_similarity = torch.mm(
            torch.mm(self.st_sn_alignment_norm, self.sn_sn_similarity), 
            self.sn_st_alignment_norm
        )
        expected_sn_sn_similarity = torch.mm(
            torch.mm(self.sn_st_alignment_norm, st_st_similarity), 
            self.st_sn_alignment_norm
        )
        
        # Compute cosine similarity between expected and actual ST-ST similarities
        expected_flat_st = expected_st_st_similarity.flatten()
        actual_flat_st = st_st_similarity.flatten()
        expected_flat_sn = expected_sn_sn_similarity.flatten()
        
        # Normalize vectors
        expected_norm_st = F.normalize(expected_flat_st.unsqueeze(0), p=2, dim=1)
        actual_norm_st = F.normalize(actual_flat_st.unsqueeze(0), p=2, dim=1)
        expected_norm_sn = F.normalize(expected_flat_sn.unsqueeze(0), p=2, dim=1)
        
        # Compute cosine similarity (we want to maximize this, so minimize 1 - similarity)
        cosine_sim_st = F.cosine_similarity(expected_norm_st, actual_norm_st, dim=1)
        similarity_loss_st = 1 - cosine_sim_st.mean()
        cosine_sim_sn = F.cosine_similarity(expected_norm_sn, self.actual_norm_sn, dim=1)
        similarity_loss_sn = 1 - cosine_sim_sn.mean()

        return similarity_loss_st, similarity_loss_sn, {
            'sn_st_alignment': self.sn_st_alignment,
            'expected_st_st_similarity': expected_st_st_similarity,
            'actual_st_st_similarity': st_st_similarity,
            'cosine_similarity_st': cosine_sim_st.mean(),
            'cosine_similarity_sn': cosine_sim_sn.mean()
        }
    
    def training_step(self, batch, batch_idx):
        # Early stopping check for stage 2
        if self.stage2_stopped:
            return None
            
        optimizer_g, optimizer_d = self.optimizers()

        x_original, edge_index, bias, alignment_matrix = batch.x, batch.edge_index, batch.bias, batch.alignment_matrix
        mask = torch.rand_like(x_original) > 0.3
        x = x_original * mask.to(x_original.device)

        if self.sn_indices is None:
            sn_size = bias
            st_size = x.size(0) - bias
            self.setup_indices(sn_size, st_size)

        x_hat = self(x, edge_index)

        loss_sn = F.mse_loss(x_hat[self.sn_indices], x[self.sn_indices])
        loss_st = F.mse_loss(
            x_hat[self.st_indices, :self.hparams.n_matching_genes],
            x[self.st_indices, :self.hparams.n_matching_genes]
        )
        imputation_loss = loss_sn + loss_st

        is_stage1 = (self.current_epoch < self.stage1_epochs) and (not self.stage1_complete)

        similarity_loss_st = torch.tensor(0.0, device=x.device)
        similarity_loss_sn = torch.tensor(0.0, device=x.device)
        similarity_info = None
        if not is_stage1:
            sn_data = x[self.sn_indices]
            st_data = x_hat[self.st_indices]
            similarity_loss_st, similarity_loss_sn, similarity_info = self.compute_similarity_loss(sn_data, st_data, alignment_matrix)

        base_total_loss = imputation_loss
        if not is_stage1:
            base_total_loss = base_total_loss + self.similarity_weight * (similarity_loss_st + similarity_loss_sn)

        if is_stage1:
            stage1_metric = base_total_loss
            if stage1_metric < self.stage1_best_loss - self.stage1_min_delta:
                self.stage1_best_loss = stage1_metric.item()
                self.stage1_wait = 0
            else:
                self.stage1_wait += 1
            if self.stage1_wait >= self.stage1_patience:
                self.stage1_stopped = True
                self.stage1_complete = True
                print(f"Early stopping triggered for Stage 1 at epoch {self.current_epoch}")
                print("Stage 1 completed early. Switching to Stage 2 with similarity regularization.")
                self.trainer.save_checkpoint('stage1_early_stopped_model.ckpt')
                self.current_epoch_stage = self.stage1_epochs
            self.log('stage1_loss', stage1_metric, batch_size=1, prog_bar=True)
            self.log('loss_sn', loss_sn, batch_size=1, prog_bar=True)
            self.log('loss_st', loss_st, batch_size=1, prog_bar=True)
            self.log('stage1_wait', self.stage1_wait, batch_size=1)
            self.log('stage1_best_loss', self.stage1_best_loss, batch_size=1)
        else:
            stage2_metric = base_total_loss
            if similarity_info is not None:
                self.log('similarity_loss_st', similarity_loss_st, batch_size=1, prog_bar=True)
                self.log('similarity_loss_sn', similarity_loss_sn, batch_size=1, prog_bar=True)
                self.log('cosine_similarity_st', similarity_info['cosine_similarity_st'], batch_size=1)
                self.log('cosine_similarity_sn', similarity_info['cosine_similarity_sn'], batch_size=1)

            self.log('stage2_base_loss', stage2_metric, batch_size=1, prog_bar=True)
            self.log('imputation_loss', imputation_loss, batch_size=1, prog_bar=True)
            self.log('loss_sn', loss_sn, batch_size=1, prog_bar=True)
            self.log('loss_st', loss_st, batch_size=1, prog_bar=True)

        if self.sn_genegraph is None:
            self.sn_genegraph = cross_dist(
                x[self.sn_indices, :self.hparams.n_matching_genes],
                x[self.sn_indices, self.hparams.n_matching_genes:]
            )
            self.sn_genegraph = self.sn_genegraph.detach()
        st_genegraph = cross_dist(
            x[self.st_indices, :self.hparams.n_matching_genes],
            x_hat[self.st_indices, self.hparams.n_matching_genes:]
        )
        loss_genegraph = self.genegraph_loss(st_genegraph, self.sn_genegraph)
        total_loss = base_total_loss + self.lam_genegraph * loss_genegraph
        self.log('loss_genegraph', loss_genegraph, batch_size=1, prog_bar=True)

        if is_stage1:
            self.toggle_optimizer(optimizer_g)
            optimizer_g.zero_grad()
            self.manual_backward(total_loss)
            optimizer_g.step()
            self.untoggle_optimizer(optimizer_g)
            return total_loss

        if self.stage2_stopped:
            self.log('stage2_loss', total_loss, batch_size=1, prog_bar=True)
            return total_loss

        real_st = x_original#[self.st_indices, :self.hparams.n_matching_genes]
        fake_st = x_hat#[self.st_indices, :self.hparams.n_matching_genes]
        fake_detached = fake_st.detach()
        d_loss = None
        if real_st.numel() > 0:
            real_detached = real_st.detach()
            for step_idx in range(self.discriminator_steps):
                self.toggle_optimizer(optimizer_d)
                optimizer_d.zero_grad()
                d_loss = self._discriminator_loss(real_detached, fake_detached)
                self.manual_backward(d_loss)
                optimizer_d.step()
                self.untoggle_optimizer(optimizer_d)

        generator_loss = total_loss
        g_adv_loss = None
        if real_st.numel() > 0:
            fake_logits = self.discriminator(fake_st)
            g_adv_loss = self.adv_loss_fn(fake_logits, torch.ones_like(fake_logits))
            generator_loss = generator_loss + self.hparams.adv_weight * g_adv_loss
        if generator_loss < self.stage2_best_loss - self.stage2_min_delta:
            self.stage2_best_loss = generator_loss.item()
            self.stage2_wait = 0
        else:
            self.stage2_wait += 1
        if self.stage2_wait >= self.stage2_patience:
            self.stage2_stopped = True
            print(f"Early stopping triggered for Stage 2 at epoch {self.current_epoch}")
            self.trainer.save_checkpoint('stage2_final_model.ckpt')
            self.trainer.should_stop = True
        self.log('stage2_wait', self.stage2_wait, batch_size=1)
        self.log('stage2_best_loss', self.stage2_best_loss, batch_size=1)

        for step_idx in range(self.generator_steps):
            self.toggle_optimizer(optimizer_g)
            optimizer_g.zero_grad()
            retain_graph = step_idx < self.generator_steps - 1
            self.manual_backward(generator_loss, retain_graph=retain_graph)
            optimizer_g.step()
            self.untoggle_optimizer(optimizer_g)

        if d_loss is not None:
            self.log('d_loss', d_loss, batch_size=1, prog_bar=True)
        if g_adv_loss is not None:
            self.log('g_adv_loss', g_adv_loss, batch_size=1, prog_bar=True)
        self.log('stage2_loss', generator_loss, batch_size=1, prog_bar=True)

        return generator_loss

    def _discriminator_loss(self, real_samples, fake_samples):
        real_logits = self.discriminator(real_samples)
        fake_logits = self.discriminator(fake_samples)
        loss_real = self.adv_loss_fn(real_logits, torch.ones_like(real_logits))
        loss_fake = self.adv_loss_fn(fake_logits, torch.zeros_like(fake_logits))
        return 0.5 * (loss_real + loss_fake)

    def configure_optimizers(self):
        generator_optimizer = torch.optim.Adam(self.imputer.parameters(), lr=self.hparams.lr)
        discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.hparams.discriminator_lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            generator_optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        return (
            [generator_optimizer, discriminator_optimizer],
            [{"scheduler": scheduler, "monitor": "stage1_loss"}],
        )

    def lr_scheduler_step(self, scheduler, optimizer_idx):
        if optimizer_idx != 0:
            return
        monitor_metric = "stage1_loss" if not self.stage1_complete else "stage2_loss"
        metric_value = self.trainer.callback_metrics.get(monitor_metric)
        if metric_value is not None:
            scheduler.step(metric_value)
        else:
            print(f"Warning: Metric '{monitor_metric}' not found. Skipping scheduler step.")
            
class TwoStageGNNImputer(L.LightningModule):
    def __init__(self, num_features, n_matching_genes, hidden_channels=64, num_layers=3, 
                 layer_type='ClusterGCN', dropout=0.3,
                 gene_masking_percent=0.3, lr=5e-4, stage1_epochs=500, 
                 similarity_weight=0.5, alignment_lr=5e-4, alignment_devices=None,
                 stage2_patience=20, stage2_min_delta=1e-4,
                 stage1_patience=10, stage1_min_delta=1e-4, lam_genegraph=0.1,
                 discriminator_hidden=256, discriminator_lr=5e-4, adv_weight=0.05,
                 discriminator_steps=1, generator_steps=1, alignment_matrix=None,
                 triplet_margin=1.0, triplet_weight=0.1, stage1_only=False):
        super().__init__()
        self.save_hyperparameters(ignore=["alignment_matrix"])
        if alignment_matrix is not None:
            # Keep sparse matrices sparse to save memory
            if sp.issparse(alignment_matrix):
                alignment_matrix = alignment_matrix.tocoo()
                indices = torch.from_numpy(np.vstack((alignment_matrix.row, alignment_matrix.col)).astype(np.int64))
                values = torch.from_numpy(alignment_matrix.data.astype(np.float32))
                shape = torch.Size(alignment_matrix.shape)
                cpu_matrix = torch.sparse_coo_tensor(indices, values, shape)
            else:
                cpu_matrix = torch.as_tensor(alignment_matrix, dtype=torch.float32)
            self.register_buffer("alignment_matrix_cpu", cpu_matrix, persistent=False)
        else:
            self.register_buffer("alignment_matrix_cpu", None, persistent=False)

        # Main imputation model
        self.imputer = GNNImputer(num_features=num_features, n_matching_genes=n_matching_genes, hidden_channels=hidden_channels, 
                                  num_layers=num_layers, layer_type=layer_type, dropout=dropout, learning_rate=lr)

        # Two-stage training parameters
        self.stage1_epochs = stage1_epochs
        self.similarity_weight = similarity_weight
        self.gene_masking_percent = gene_masking_percent
        self.alignment_lr = alignment_lr
        self.alignment_devices = alignment_devices if alignment_devices is not None else [0]
        self.current_epoch_stage = 0
        self.stage1_only = stage1_only

        # Early stopping parameters for stage 1
        self.stage1_patience = stage1_patience
        self.stage1_min_delta = stage1_min_delta
        self.stage1_best_loss = float('inf')
        self.stage1_wait = 0

        # Early stopping parameters for stage 2
        self.stage2_patience = stage2_patience
        self.stage2_min_delta = stage2_min_delta
        self.stage2_best_loss = float('inf')
        self.stage2_wait = 0
        self.stage2_stopped = False

        # For storing intermediate results
        self.stage1_complete = False
        self.sn_indices = None
        self.st_indices = None
        self.sn_sn_similarity = None
        self.lam_genegraph = lam_genegraph
        self.genegraph_loss = CosineLoss()  # Initialize the gene graph loss
        self.sn_genegraph = None  # Cache for SN_genegraph
        self.automatic_optimization = False
        disc_in_features = max(1, int(self.hparams.n_matching_genes))
        hidden_mid = max(1, discriminator_hidden // 2)
        self.discriminator = nn.Sequential(
            nn.Linear(disc_in_features, discriminator_hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(discriminator_hidden, hidden_mid),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_mid, 1)
        )
        self.adv_loss_fn = nn.BCEWithLogitsLoss()
        self.generator_steps = generator_steps
        self.discriminator_steps = discriminator_steps
        self.triplet_loss_fn = nn.TripletMarginLoss(margin=triplet_margin)
        self.triplet_weight = triplet_weight

    def setup_indices(self, sn_size, st_size):
        """Setup indices for SN and ST data"""
        sn_size = int(sn_size)
        st_size = int(st_size)
        if sn_size <= 0 or st_size <= 0:
            raise ValueError("sn_size and st_size must be positive integers.")
        self.sn_indices = torch.arange(sn_size)
        self.st_indices = torch.arange(sn_size, sn_size + st_size)

    def forward(self, x, edge_index):
        return self.imputer(x, edge_index)

    def compute_alignment_matrices(self, sn_data, st_data):
        """Compute alignment matrices using scGALA's get_alignments function"""
        from .main import get_alignments

        # Create temporary AnnData objects
        sn_adata = ad.AnnData(X=sn_data.detach().cpu().numpy())
        st_adata = ad.AnnData(X=st_data.detach().cpu().numpy())

        # Get alignment matrix using scGALA
        # We only want edge probabilities for similarity preservation loss
        alignment_matrix = get_alignments(
            adata1=sn_adata, 
            adata2=st_adata,
            k=20,
            min_value=0.8, 
            lr=self.alignment_lr,
            max_epochs=10,  # Fewer epochs for efficiency
            get_edge_probs=True,
            get_matrix=False,  # Important: only get raw edge probs as a single sparse matrix
            scale=True,
            devices=self.alignment_devices,
            default_root_dir='./logs/scgala_alignment',
        )

        # Add explicit GPU memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Clean up temporary AnnData objects
        del sn_adata, st_adata
        gc.collect()

        print('Alignment matrix computed at epoch:', self.current_epoch_stage)

        if sp.issparse(alignment_matrix):
            alignment_matrix = alignment_matrix.tocoo()
            indices = torch.from_numpy(np.vstack((alignment_matrix.row, alignment_matrix.col)).astype(np.int64))
            values = torch.from_numpy(alignment_matrix.data.astype(np.float32))
            shape = torch.Size(alignment_matrix.shape)
            return torch.sparse_coo_tensor(indices, values, shape).to(self.device)
        else:
            return torch.tensor(alignment_matrix, device=self.device, dtype=torch.float32)
    @staticmethod
    def _sparse_row_normalize(sparse_tensor):
        """Row-normalize a sparse or dense tensor"""
        if not sparse_tensor.is_sparse:
            return F.normalize(sparse_tensor, p=1, dim=1)

        # For COO sparse tensors
        sparse_tensor = sparse_tensor.coalesce()
        indices = sparse_tensor.indices()
        values = sparse_tensor.values()

        # Compute row sums
        row_sums = torch.zeros(sparse_tensor.size(0), device=sparse_tensor.device)
        row_sums.scatter_add_(0, indices[0], values)

        # Divide by row sums
        normalized_values = values / row_sums[indices[0]].clamp(min=EPS)

        return torch.sparse_coo_tensor(indices, normalized_values, sparse_tensor.size())

    def compute_similarity_matrices(self, data, k=20, sparse=True):
        """Compute sparse pairwise similarity matrices using cosine similarity with K-NN, preserving gradients."""
        # Normalize data
        data_norm = F.normalize(data, p=2, dim=1)

        # Compute full cosine similarity for finding K-NN
        full_similarity = torch.mm(data_norm, data_norm.t())

        if sparse:
            # Find top-k similarities for each node (including self)
            topk_values, topk_indices = torch.topk(full_similarity, k=k+1, dim=1, largest=True)

            # Create a mask for the K-NN values
            n_nodes = data.size(0)
            mask = torch.zeros_like(full_similarity, dtype=torch.bool)
            mask.scatter_(1, topk_indices, True)

            # Use the mask to keep only K-NN similarities, set others to zero (preserves gradient)
            similarity = torch.where(mask, full_similarity, torch.zeros_like(full_similarity))
        else:
            # Use full similarity matrix directly
            similarity = full_similarity
        # Make symmetric by taking max(sim[i,j], sim[j,i])
        similarity = torch.max(similarity, similarity.t())

        return similarity

    def compute_similarity_loss(self, sn_data, st_data):
        """Compute similarity preservation loss"""
        # Get alignment probabilities between SN and ST
        if not hasattr(self, 'st_sn_alignment_norm'):
            sn_st_alignment = self.alignment_matrix_cpu.to(self.device, non_blocking=True)
            st_sn_alignment = sn_st_alignment.t()
            # Row-normalize alignment matrices (each row sums to 1)
            self.st_sn_alignment_norm = self._sparse_row_normalize(st_sn_alignment).detach().to(self.device)
            self.sn_st_alignment_norm = self._sparse_row_normalize(sn_st_alignment).detach().to(self.device)
            del sn_st_alignment, st_sn_alignment, self.alignment_matrix_cpu  # Free memory

        # Compute similarity matrices
        if self.sn_sn_similarity is None:
            sn_sim = self.compute_similarity_matrices(sn_data, sparse=True, k=20)
            self.sn_sn_similarity_cpu = sn_sim.detach().cpu()
            self.actual_norm_sn = F.normalize(
                self.sn_sn_similarity_cpu.flatten().unsqueeze(0), p=2, dim=1
            )
        sn_sn_similarity = self.sn_sn_similarity_cpu.to(self.device, non_blocking=True)

        st_st_similarity = self.compute_similarity_matrices(st_data, sparse=False,k=20).to(self.device)

        # Expected ST-ST similarity based on SN-SN similarity and SN-ST alignment
        # expected_st_st = ST-SN @ SN-SN @ SN-ST

        if not hasattr(self, 'expected_norm_st'):
            if self.st_sn_alignment_norm.is_sparse:
                # Use sparse-dense multiplication
                temp = torch.sparse.mm(self.st_sn_alignment_norm, sn_sn_similarity)
                # (temp @ sparse) = (sparse.t() @ temp.t()).t()
                expected_st = torch.sparse.mm(self.sn_st_alignment_norm.t(), temp.t()).t()
            else:
                expected_st = torch.mm(
                    torch.mm(self.st_sn_alignment_norm, sn_sn_similarity),
                    self.sn_st_alignment_norm
                )
            self.expected_norm_st = F.normalize(
                expected_st.flatten().unsqueeze(0), p=2, dim=1
            ).cpu()

        if self.sn_st_alignment_norm.is_sparse:
            temp_sn = torch.sparse.mm(self.sn_st_alignment_norm, st_st_similarity)
            expected_sn_sn_similarity = torch.sparse.mm(self.st_sn_alignment_norm.t(), temp_sn.t()).t()
        else:
            expected_sn_sn_similarity = torch.mm(
                torch.mm(self.sn_st_alignment_norm, st_st_similarity), 
                self.st_sn_alignment_norm
            )

        # Compute cosine similarity between expected and actual ST-ST similarities

        actual_flat_st = st_st_similarity.flatten()
        expected_flat_sn = expected_sn_sn_similarity.flatten()

        # Normalize vectors
        actual_norm_st = F.normalize(actual_flat_st.unsqueeze(0), p=2, dim=1)
        expected_norm_sn = F.normalize(expected_flat_sn.unsqueeze(0), p=2, dim=1)

        expected_norm_st = self.expected_norm_st.to(self.device, non_blocking=True)
        # Compute cosine similarity (we want to maximize this, so minimize 1 - similarity)
        cosine_sim_st = F.cosine_similarity(expected_norm_st, actual_norm_st, dim=1)
        similarity_loss_st = 1 - cosine_sim_st.mean()
        cosine_sim_sn = F.cosine_similarity(expected_norm_sn, self.actual_norm_sn.to(self.device, non_blocking=True), dim=1)
        similarity_loss_sn = 1 - cosine_sim_sn.mean()

        return similarity_loss_st, similarity_loss_sn    
    def compute_triplet_loss(self, embeddings, edge_index):
        edge_index = edge_index.to(embeddings.device)
        if edge_index.numel() == 0:
            return embeddings.new_tensor(0.0)
        anchors, positives = edge_index
        mask = anchors != positives
        anchors = anchors[mask]
        positives = positives[mask]
        if anchors.numel() == 0:
            return embeddings.new_tensor(0.0)
        direction_mask = anchors < positives
        if direction_mask.any():
            anchors = anchors[direction_mask]
            positives = positives[direction_mask]
        num_nodes = embeddings.size(0)
        negatives = torch.randint(0, num_nodes, anchors.size(), device=embeddings.device)
        neg_mask = (negatives == anchors) | (negatives == positives)
        while neg_mask.any():
            negatives[neg_mask] = torch.randint(0, num_nodes, (neg_mask.sum().item(),), device=embeddings.device)
            neg_mask = (negatives == anchors) | (negatives == positives)
        return self.triplet_loss_fn(
            embeddings[anchors],
            embeddings[positives],
            embeddings[negatives],
        )
    
    def training_step(self, batch, batch_idx):
        # Early stopping check for stage 2
        if self.stage2_stopped:
            return None
            
        optimizer_g, optimizer_d = self.optimizers()

        x_original, edge_index, bias = batch.x, batch.edge_index, batch.bias
        mask = torch.rand_like(x_original[:, self.hparams.n_matching_genes:]) > self.gene_masking_percent
        x = x_original.clone()
        x[:, self.hparams.n_matching_genes:] = x[:, self.hparams.n_matching_genes:] * mask.to(x.device)

        if self.sn_indices is None:
            sn_size = bias
            st_size = x.size(0) - bias
            self.setup_indices(sn_size, st_size)

        x_hat = self(x, edge_index)

        loss_sn = F.mse_loss(x_hat[self.sn_indices], x_original[self.sn_indices])
        loss_st = F.mse_loss(
            x_hat[self.st_indices, :self.hparams.n_matching_genes],
            x_original[self.st_indices, :self.hparams.n_matching_genes]
        )
        imputation_loss = loss_sn + loss_st
        # base_total_loss = imputation_loss
        triplet_loss = torch.tensor(0.0, device=x.device)
        if self.triplet_weight > 0:
            triplet_loss = self.compute_triplet_loss(x_hat, edge_index)
        base_total_loss = imputation_loss + self.triplet_weight * triplet_loss
        self.log('triplet_loss', triplet_loss, batch_size=1, prog_bar=True)

        similarity_loss_st = torch.tensor(0.0, device=x.device)
        similarity_loss_sn = torch.tensor(0.0, device=x.device)
        
        if self.stage1_complete and self.similarity_weight > 0:
            sn_data = x_original[self.sn_indices]
            st_data = x_hat[self.st_indices]
            similarity_loss_st, similarity_loss_sn = self.compute_similarity_loss(sn_data, st_data)
            base_total_loss = base_total_loss + self.similarity_weight * (similarity_loss_st + similarity_loss_sn)
            
        if not self.stage1_complete:
            stage1_metric = base_total_loss
            if stage1_metric < self.stage1_best_loss - self.stage1_min_delta:
                self.stage1_best_loss = stage1_metric.item()
                self.stage1_wait = 0
            else:
                self.stage1_wait += 1
            if self.stage1_wait >= self.stage1_patience or self.current_epoch >= self.stage1_epochs:
                self.stage1_complete = True
                print(f"Early stopping triggered for Stage 1 at epoch {self.current_epoch}")
                print("Stage 1 completed early. Switching to Stage 2 with similarity regularization.")
                self.trainer.save_checkpoint(f'stage1_early_stopped_model_{self.hparams.layer_type}.ckpt')
                self.current_epoch_stage = self.stage1_epochs
                if self.stage1_only:
                    self.trainer.should_stop = True
            self.log('stage1_loss', stage1_metric, batch_size=1, prog_bar=True)
            self.log('loss_sn', loss_sn, batch_size=1, prog_bar=True)
            self.log('loss_st', loss_st, batch_size=1, prog_bar=True)
            self.log('stage1_wait', self.stage1_wait, batch_size=1)
            self.log('stage1_best_loss', self.stage1_best_loss, batch_size=1)
        else:
            stage2_metric = base_total_loss
            self.log('similarity_loss_st', similarity_loss_st, batch_size=1, prog_bar=True)
            self.log('similarity_loss_sn', similarity_loss_sn, batch_size=1, prog_bar=True)

            self.log('stage2_base_loss', stage2_metric, batch_size=1, prog_bar=True)
            self.log('imputation_loss', imputation_loss, batch_size=1, prog_bar=True)
            self.log('loss_sn', loss_sn, batch_size=1, prog_bar=True)
            self.log('loss_st', loss_st, batch_size=1, prog_bar=True)

        loss_genegraph = torch.tensor(0.0, device=x_original.device)
        if self.lam_genegraph > 0:
            if self.sn_genegraph is None:
                self.sn_genegraph = cross_dist(
                    x_original[self.sn_indices, :self.hparams.n_matching_genes],
                    x_original[self.sn_indices, self.hparams.n_matching_genes:]
                )
                self.sn_genegraph = self.sn_genegraph.detach()
            st_genegraph = cross_dist(
                x_hat[self.st_indices, :self.hparams.n_matching_genes],
                x_hat[self.st_indices, self.hparams.n_matching_genes:]
            )
            loss_genegraph = self.genegraph_loss(st_genegraph, self.sn_genegraph)
        total_loss = base_total_loss + self.lam_genegraph * loss_genegraph
        self.log('loss_genegraph', loss_genegraph, batch_size=1, prog_bar=True)

        # GAN training now active in both stages
        real_st = x_original[self.st_indices, :self.hparams.n_matching_genes]
        fake_st = x_hat[self.st_indices, :self.hparams.n_matching_genes]
        # real_st = x_original[self.sn_indices, self.hparams.n_matching_genes:]
        # fake_st = x_hat[self.st_indices,self.hparams.n_matching_genes:]
        fake_detached = fake_st.detach()
        d_loss = None
        if real_st.numel() > 0:
            real_detached = real_st.detach()
            for step_idx in range(self.discriminator_steps):
                self.toggle_optimizer(optimizer_d)
                optimizer_d.zero_grad()
                d_loss = self._discriminator_loss(real_detached, fake_detached)
                self.manual_backward(d_loss)
                optimizer_d.step()
                self.untoggle_optimizer(optimizer_d)

        generator_loss = total_loss
        g_adv_loss = torch.tensor(0.0, device=x_original.device)
        if real_st.numel() > 0 and self.hparams.adv_weight > 0:
            fake_logits = self.discriminator(fake_st)
            g_adv_loss = self.adv_loss_fn(fake_logits, torch.ones_like(fake_logits))
            generator_loss = generator_loss + self.hparams.adv_weight * g_adv_loss
        
        if not self.stage1_complete:
            self.toggle_optimizer(optimizer_g)
            optimizer_g.zero_grad()
            self.manual_backward(generator_loss)
            optimizer_g.step()
            self.untoggle_optimizer(optimizer_g)
            if d_loss is not None:
                self.log('d_loss', d_loss, batch_size=1, prog_bar=True)
            if g_adv_loss is not None:
                self.log('g_adv_loss', g_adv_loss, batch_size=1, prog_bar=True)
            return generator_loss

        if self.stage2_stopped:
            self.log('stage2_loss', generator_loss, batch_size=1, prog_bar=True)
            return generator_loss

        if generator_loss < self.stage2_best_loss - self.stage2_min_delta:
            self.stage2_best_loss = generator_loss.item()
            self.stage2_wait = 0
        else:
            self.stage2_wait += 1
        if self.stage2_wait >= self.stage2_patience:
            self.stage2_stopped = True
            print(f"Early stopping triggered for Stage 2 at epoch {self.current_epoch}")
            self.trainer.save_checkpoint(f'stage2_final_model_{self.hparams.layer_type}.ckpt')
            self.trainer.should_stop = True
       
        self.log('stage2_wait', self.stage2_wait, batch_size=1)
        self.log('stage2_best_loss', self.stage2_best_loss, batch_size=1)

        for step_idx in range(self.generator_steps):
            self.toggle_optimizer(optimizer_g)
            optimizer_g.zero_grad()
            retain_graph = step_idx < self.generator_steps - 1
            self.manual_backward(generator_loss, retain_graph=retain_graph)
            optimizer_g.step()
            self.untoggle_optimizer(optimizer_g)

        if d_loss is not None:
            self.log('d_loss', d_loss, batch_size=1, prog_bar=True)
        if g_adv_loss is not None:
            self.log('g_adv_loss', g_adv_loss, batch_size=1, prog_bar=True)
        self.log('stage2_loss', generator_loss, batch_size=1, prog_bar=True)

        return generator_loss

    def _discriminator_loss(self, real_samples, fake_samples):
        real_logits = self.discriminator(real_samples)
        fake_logits = self.discriminator(fake_samples)
        loss_real = self.adv_loss_fn(real_logits, torch.ones_like(real_logits))
        loss_fake = self.adv_loss_fn(fake_logits, torch.zeros_like(fake_logits))
        return 0.5 * (loss_real + loss_fake)

    def configure_optimizers(self):
        generator_optimizer = torch.optim.Adam(self.imputer.parameters(), lr=self.hparams.lr)
        discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.hparams.discriminator_lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            generator_optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        return (
            [generator_optimizer, discriminator_optimizer],
            [{"scheduler": scheduler, "monitor": "stage1_loss"}],
        )

    def lr_scheduler_step(self, scheduler, optimizer_idx):
        if optimizer_idx != 0:
            return
        monitor_metric = "stage1_loss" if not self.stage1_complete else "stage2_loss"
        metric_value = self.trainer.callback_metrics.get(monitor_metric)
        if metric_value is not None:
            scheduler.step(metric_value)
        else:
            print(f"Warning: Metric '{monitor_metric}' not found. Skipping scheduler step.")