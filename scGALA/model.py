# For Cell Alignment
from typing import Any, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import pytorch_lightning as pl
from torch_geometric.nn import (
    GCNConv,
    GATConv,
    GATv2Conv,
    ClusterGCNConv,
    AGNNConv,
    VGAE,
    EGConv,
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
import GCL.augmentors as A
import anndata as ad
import numpy as np
import gc

# Constants
EPS = 1e-15
MAX_LOGSTD = 10

class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels=-1, out_channels=256, dropout = 0.3):
        super(VariationalGCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True) # cached only for transductive learning
        self.conv_mu = GCNConv(2 * out_channels, out_channels, cached=True)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels, cached=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(self.dropout(x))
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

aug = A.RandomChoice([
                    A.FeatureMasking(pf=0.1),
                    A.EdgeRemoving(pe=0.3)],
                    num_choices=1)  ##A.NodeDropping(pn=0.1),

class VGAE_gcl(L.LightningModule):
    def __init__(self,in_channels:int = -1,out_channels:int = 256,dropout:float=0.3,lr:float=3e-4,use_scheduler:bool = True,optimizer:Literal['adam','sgd'] = 'adam') -> None:
        super().__init__()
        # self.x, self.edge_index, self.edge_weight, self.data = get_graph(data1,data2,k)
        self.lr = lr
        self.model = VGAE(VariationalGCNEncoder(in_channels=in_channels,out_channels=out_channels, dropout=dropout)) 
        self.use_scheduler = use_scheduler
        self.optimizer = optimizer
    def training_step(self, batch, batch_idx):
        x, edge_index, bias, num_nodes = batch
        x, edge_index, bias, num_nodes = x[0], edge_index[0], bias[0], num_nodes[0]
        # pdb.set_trace()
        x_new, edge_index_new,_ = aug(x, edge_index)
        z = self.model.encode(x_new,edge_index_new)
        vae_loss = self.model.recon_loss(z, edge_index) 
        vae_loss = vae_loss + (1 / num_nodes) * self.model.kl_loss()  # new line
        self.log_dict({'train_loss':float(vae_loss)},prog_bar=True)
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
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.2,patience=2,threshold=5e-2,    threshold_mode='rel',verbose=True),
                "monitor": 'ave_align',
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
        x, edge_index, bias, num_nodes = batch
        x, edge_index, bias, num_nodes = x[0], edge_index[0], bias[0], num_nodes[0]
        z = self.model.encode(x,edge_index)
        neg_edge_index = negative_sampling(edge_index, z.size(0))
        auc,ap = self.model.test(z, edge_index, neg_edge_index)
        likelyhood = torch.cat([(z[i]*z).sum(dim=1).unsqueeze(0) for i in range(bias)],dim=0)[:bias,bias:].sigmoid()
        self.log_dict({'auc':auc,'ap':ap,'ave_align':likelyhood[likelyhood>0.9].shape[0]/likelyhood.shape[0]},prog_bar=True)
    # def forward(self,x,edge_index) -> Any:
    #     return self.model.encode(x,edge_index)
    def predict_step(self,batch, batch_idx) -> Any:
        x, edge_index, bias, num_nodes = batch
        x, edge_index, bias, num_nodes = x[0], edge_index[0], bias[0], num_nodes[0]
        z = self.model.encode(x,edge_index)
        return z

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
    
class MSVGAE_gcl(L.LightningModule):
    def __init__(self,in_channels:int = -1,out_channels:list = [16,32,64],out_dim=64,dropout:float=0.3,lr:float=3e-4,use_scheduler:bool = True,optimizer:Literal['adam','sgd'] = 'adam',version = 'normal') -> None:
        super().__init__()
        # self.x, self.edge_index, self.edge_weight, self.data = get_graph(data1,data2,k)
        self.lr = lr
        if version == 'normal':
            self.model = MSVGAE(nn.ModuleList([GAT_Encoder(num_heads=[4,4,4,4],in_channels=in_channels,latent_dim=out_channels[i], dropout=dropout) for i in range(len(out_channels))]),out_dim=out_dim)
        if version == 'simple':
            self.model = MSVGAE(nn.ModuleList([GAT_Encoder(num_heads=[1,1,1,1],in_channels=in_channels,latent_dim=out_channels[i], dropout=dropout) for i in range(len(out_channels))]),out_dim=out_dim)
        if version == 'naive':
            self.model = MSVGAE(nn.ModuleList([GAT_Encoder_no_hidden(num_heads=[1,1,1,1],in_channels=in_channels,latent_dim=out_channels[i], dropout=dropout) for i in range(len(out_channels))]),out_dim=out_dim)
        self.use_scheduler = use_scheduler
        self.optimizer = optimizer
    def training_step(self, batch, batch_idx):
        x, edge_index, bias, num_nodes = batch
        x, edge_index, bias, num_nodes = x[0], edge_index[0], bias[0], num_nodes[0]
        # pdb.set_trace()
        x_new, edge_index_new,_ = aug(x, edge_index)
        
        z = self.model.encode(x_new,edge_index_new)
        
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
        x, edge_index, bias, num_nodes = batch
        x, edge_index, bias, num_nodes = x[0], edge_index[0], bias[0], num_nodes[0]
        z = self.model.encode(x,edge_index)
        neg_edge_index = negative_sampling(edge_index, z.size(0))
        auc,ap = self.model.test(z, edge_index, neg_edge_index)
        likelyhood = torch.matmul(z[:bias], z[bias:].T).sigmoid()
        self.log_dict({'auc':auc,'ap':ap,'ave_align':likelyhood[likelyhood>0.9].shape[0]/likelyhood.shape[0]},prog_bar=True)
    # def forward(self,x,edge_index) -> Any:
    #     return self.model.encode(x,edge_index)
    def predict_step(self,batch, batch_idx) -> Any:
        x, edge_index, bias, num_nodes = batch
        x, edge_index, bias, num_nodes = x[0], edge_index[0], bias[0], num_nodes[0]
        
        # with Path(r'./Results/latent') as latent_dir:
        #     if not latent_dir.exists():
        #         latent_dir.mkdir(parents=True)
        z = self.model.encode(x,edge_index)
        # torch.save(z.detach().cpu(),'Results/latent/'+str(self.current_epoch)+'.pt')
        return z

class MSVGAE_gcl_spatialGW(L.LightningModule):
    def __init__(self,in_channels:int = -1,out_channels:list = [16,32,64],out_dim=64,dropout:float=0.3,lr:float=3e-4,use_scheduler:bool = True,optimizer:Literal['adam','sgd'] = 'adam',version = 'normal') -> None:
        super().__init__()
        # self.x, self.edge_index, self.edge_weight, self.data = get_graph(data1,data2,k)
        self.lr = lr
        if version == 'normal':
            self.model = MSVGAE(nn.ModuleList([GAT_Encoder(num_heads=[4,4,4,4],in_channels=in_channels,latent_dim=out_channels[i], dropout=dropout) for i in range(len(out_channels))]),out_dim=out_dim)
        if version == 'simple':
            self.model = MSVGAE(nn.ModuleList([GAT_Encoder(num_heads=[1,1,1,1],in_channels=in_channels,latent_dim=out_channels[i], dropout=dropout) for i in range(len(out_channels))]),out_dim=out_dim)
        if version == 'naive':
            self.model = MSVGAE(nn.ModuleList([GAT_Encoder_no_hidden(num_heads=[1,1,1,1],in_channels=in_channels,latent_dim=out_channels[i], dropout=dropout) for i in range(len(out_channels))]),out_dim=out_dim)
        self.use_scheduler = use_scheduler
        self.optimizer = optimizer    
        try:
            import ot
            self.ot = ot
        except ImportError:
            raise ImportError('\nplease install pot:\n\tpip install POT')
        # Define the triplet margin loss criterion
        self.criterion = nn.TripletMarginLoss(margin=1.0)
    def training_step(self, batch, batch_idx):
        x, edge_index, bias, num_nodes, C1, C2, spatial_edge_index = batch
        x, edge_index, bias, num_nodes, C1, C2, spatial_edge_index = x[0], edge_index[0], bias[0], num_nodes[0], C1[0], C2[0],spatial_edge_index[0]
        # pdb.set_trace()
        x_new, edge_index_new,_ = aug(x, edge_index)
        
        z = self.model.encode(x_new,edge_index_new)
        
        vae_loss = self.model.recon_loss(z, edge_index) 
        # spatial_recons_loss = self.model.recon_loss(z, spatial_edge_index)*2
        # reconstructed_features = self.model.liner_decoder(z)
        # decoder_loss = torch.nn.functional.mse_loss(reconstructed_features, x) #* 10
        total_loss = vae_loss + (1 / num_nodes) * self.model.kl_loss() # new line
        
        # # Gromov-Wasserstein distance
        # likelyhood = torch.matmul(z[:bias], z[bias:].T).softmax(dim=1)
        # if self.current_epoch==0 and batch_idx == 0:
        #     p = self.ot.utils.unif(C1.shape[0], type_as=C1)
        #     q = self.ot.utils.unif(C2.shape[0], type_as=C1)
        #     self.consC, self.hC1, self.hC2 = self.ot.gromov.init_matrix(C1, C2, p, q)
        #     self.consC, self.hC1, self.hC2 = self.consC.float(), self.hC1.float(), self.hC2.float()
        # # print(self.consC.dtype,self.hC1.dtype,self.hC2.dtype,likelyhood.dtype)
        # # print(self.consC.min(),self.hC1.min(),self.hC2.min(),likelyhood.min(),C1.min(),C2.min())
        # if self.current_epoch <1:
        #     gwloss = 0
        # else:
        #     gwloss = self.ot.gromov.gwloss(self.consC, self.hC1, self.hC2, likelyhood)/(C1.shape[0]*C2.shape[0])
        spatial_triplet_loss = self.compute_triplet_loss(z, spatial_edge_index)*6
        total_loss = total_loss + spatial_triplet_loss
        self.log_dict({'total_loss':float(total_loss),'vae_loss':float(vae_loss),'spatial_triplet_loss':float(spatial_triplet_loss)},prog_bar=True)
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
        x, edge_index, bias, num_nodes, C1, C2,spatial_edge_index = batch
        x, edge_index, bias, num_nodes, C1, C2,spatial_edge_index = x[0], edge_index[0], bias[0], num_nodes[0], C1[0], C2[0],spatial_edge_index[0]
        z = self.model.encode(x,edge_index)
        neg_edge_index = negative_sampling(edge_index, z.size(0))
        auc,ap = self.model.test(z, edge_index, neg_edge_index)
        likelyhood = torch.matmul(z[:bias], z[bias:].T).sigmoid()
        self.log_dict({'auc':auc,'ap':ap,'ave_align':likelyhood[likelyhood>0.9].shape[0]/likelyhood.shape[0]},prog_bar=True)
    # def forward(self,x,edge_index) -> Any:
    #     return self.model.encode(x,edge_index)
    def predict_step(self,batch, batch_idx) -> Any:
        x, edge_index, bias, num_nodes, C1, C2,spatial_edge_index = batch
        x, edge_index = x[0], edge_index[0]
        z = self.model.encode(x,edge_index)
        # torch.save(z.detach().cpu(),'Results/latent/'+str(self.current_epoch)+'.pt')
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
class TwoStageGNNImputer(L.LightningModule):
    def __init__(self, num_features, n_matching_genes, hidden_channels=512, num_layers=3, 
                 layer_type='ClusterGCN', dropout=0.3, lr=1e-3, stage1_epochs=500, 
                 similarity_weight=0.5, alignment_lr=1e-4, alignment_devices=None,
                 alignment_update_freq_delta=50,stage2_patience=20, stage2_min_delta=1e-4):
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
    
    def compute_similarity_matrices(self, data, k=10, sparse=True):
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
        
        # Compute similarity matrices
        if self.sn_sn_similarity is None:
            self.sn_sn_similarity = self.compute_similarity_matrices(sn_data,sparse=True,k=20)
        st_st_similarity = self.compute_similarity_matrices(st_data, sparse=False)
        
        # Expected ST-ST similarity based on SN-SN similarity and SN-ST alignment
        # expected_st_st = ST-SN @ SN-SN @ SN-ST
        st_sn_alignment = self.sn_st_alignment.t()  # Transpose to get ST-SN
        # Row-normalize alignment matrices (each row sums to 1)
        st_sn_alignment_norm = F.normalize(st_sn_alignment, p=1, dim=1)
        sn_st_alignment_norm = F.normalize(self.sn_st_alignment, p=1, dim=1)
        expected_st_st_similarity = torch.mm(
            torch.mm(st_sn_alignment_norm, self.sn_sn_similarity), 
            sn_st_alignment_norm
        )
        
        # Compute cosine similarity between expected and actual ST-ST similarities
        expected_flat = expected_st_st_similarity.flatten()
        actual_flat = st_st_similarity.flatten()
        
        # Normalize vectors
        expected_norm = F.normalize(expected_flat.unsqueeze(0), p=2, dim=1)
        actual_norm = F.normalize(actual_flat.unsqueeze(0), p=2, dim=1)
        
        # Compute cosine similarity (we want to maximize this, so minimize 1 - similarity)
        cosine_sim = F.cosine_similarity(expected_norm, actual_norm, dim=1)
        similarity_loss = 1 - cosine_sim.mean()
        
        return similarity_loss, {
            'sn_st_alignment': self.sn_st_alignment,
            'expected_st_st_similarity': expected_st_st_similarity,
            'actual_st_st_similarity': st_st_similarity,
            'cosine_similarity': cosine_sim.mean()
        }
    
    def training_step(self, batch, batch_idx):
        # Early stopping check for stage 2
        if self.stage2_stopped:
            return None
            
        x, edge_index, bias = batch.x, batch.edge_index, batch.bias
        
        # Setup indices if not done
        if self.sn_indices is None:
            sn_size = bias
            st_size = x.size(0) - bias
            self.setup_indices(sn_size, st_size)
        
        # Forward pass
        x_hat = self(x, edge_index)
        
        # Stage 1: Regular imputation loss
        loss_sn = F.mse_loss(x_hat[self.sn_indices], x[self.sn_indices])
        loss_st = F.mse_loss(x_hat[self.st_indices, :self.hparams.n_matching_genes], 
                            x[self.st_indices, :self.hparams.n_matching_genes])
        imputation_loss = loss_sn + loss_st
        
        # Determine current stage based on both epoch count and stage1_complete flag
        is_stage1 = (self.current_epoch < self.stage1_epochs) and (not self.stage1_complete)
        
        if is_stage1:
            # Stage 1: Only imputation loss
            total_loss = imputation_loss
            self.log('stage1_loss', total_loss, batch_size=1, prog_bar=True)
            self.log('loss_sn', loss_sn, batch_size=1, prog_bar=True)
            self.log('loss_st', loss_st, batch_size=1, prog_bar=True)
        else:
            # Stage 2: Imputation + similarity preservation loss
            sn_data = x[self.sn_indices]
            st_data = x_hat[self.st_indices]
            
            # Compute similarity loss with proper gradient flow
            similarity_loss, similarity_info = self.compute_similarity_loss(sn_data, st_data)
            
            # Combined loss
            total_loss = imputation_loss + self.similarity_weight * similarity_loss
            
            # Early stopping logic for stage 2
            if total_loss < self.stage2_best_loss - self.stage2_min_delta:
                self.stage2_best_loss = total_loss.item()
                self.stage2_wait = 0
            else:
                self.stage2_wait += 1
                
            if self.stage2_wait >= self.stage2_patience:
                self.stage2_stopped = True
                print(f"Early stopping triggered for Stage 2 at epoch {self.current_epoch}")
                # Save the final model
                self.trainer.save_checkpoint('stage2_final_model.ckpt')
                # Signal the trainer to stop
                self.trainer.should_stop = True
            
            # Logging
            self.log('stage2_loss', total_loss, batch_size=1, prog_bar=True)
            self.log('imputation_loss', imputation_loss, batch_size=1, prog_bar=True)
            self.log('similarity_loss', similarity_loss, batch_size=1, prog_bar=True)
            self.log('cosine_similarity', similarity_info['cosine_similarity'], batch_size=1)
            self.log('loss_sn', loss_sn, batch_size=1, prog_bar=True)
            self.log('loss_st', loss_st, batch_size=1, prog_bar=True)
            self.log('stage2_wait', self.stage2_wait, batch_size=1)
            self.log('stage2_best_loss', self.stage2_best_loss, batch_size=1)
        
        return total_loss
    
    def on_train_epoch_end(self):
        self.current_epoch_stage += 1
        
        # Mark stage 1 completion (only if not already complete from checkpoint loading)
        if self.current_epoch_stage == self.stage1_epochs and not self.stage1_complete:
            self.stage1_complete = True
            print(f"Stage 1 completed. Switching to Stage 2 with similarity regularization.")
            print('save the stage 1 model')
            self.trainer.save_checkpoint('stage1_model.ckpt')
            self.imputer.eval()
            with torch.no_grad():
                # Save the imputed data after stage 1
                x_hat = self.imputer(self.trainer.datamodule.data.x.to(self.device), 
                                     self.trainer.datamodule.data.edge_index.to(self.device))
                # Extract imputed spatial data
                imputed_st = x_hat[self.trainer.datamodule.data.bias:].cpu().numpy()
                var_names = np.loadtxt('./var_names_two_stage.txt',dtype=str)[1:].tolist()
                # Create new AnnData with imputed results
                adata_st_imputed = ad.AnnData(imputed_st)
                adata_st_imputed.var_names = var_names
                # Save the imputed AnnData object
                adata_st_imputed.write('./adata_st_imputed_first_stage.h5ad')
                del adata_st_imputed, imputed_st, x_hat
                print("Imputed ST data (Stage 1) saved to './adata_st_imputed_first_stage.h5ad'")
            self.imputer.train()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "stage1_loss" if not self.stage1_complete else "stage2_loss",
            },
        }

    def lr_scheduler_step(self, scheduler, optimizer_idx):
        # Dynamically select the metric to monitor
        monitor_metric = "stage1_loss" if not self.stage1_complete else "stage2_loss"
        metric_value = self.trainer.callback_metrics.get(monitor_metric)
        if metric_value is not None:
            scheduler.step(metric_value)
        else:
            print(f"Warning: Metric '{monitor_metric}' not found. Skipping scheduler step.")