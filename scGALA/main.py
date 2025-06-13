import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import warnings
import scanpy as sc
import gc
from .model import MSVGAE_gcl,MSVGAE_gcl_spatialGW,TwoStageGNNImputer
from .data import MyDataModule,TwoStageDataModule
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping,ModelSummary,ModelCheckpoint
from typing import Literal
import pandas as pd
from .utils import make_alignments,find_mutual_nn
import torch
import numpy as np
import scipy.sparse as sp
import time
from .utils import nn_approx,nn,compute_anchor_score

from anndata import AnnData
warnings.filterwarnings('ignore', '.*deprecated.*')

torch.set_float32_matmul_precision('medium')
EPS = 1e-15

def get_alignments(data1_dir=None, data2_dir=None,adata1=None,adata2=None, out_dim:int = 32, dropout:float = 0.3, lr:float = 1e-3,min_epochs:int = 10, k:int =20, min_value=0.9, default_root_dir=None,max_epochs:int = 30,lamb = 0.3,ckpt_dir = None, transformed = False, transformed_datas=None, use_scheduler:bool = True,optimizer:Literal['adam','sgd'] = 'adam',get_latent:bool = False, get_edge_probs:bool = False, get_matrix:bool = True,only_mnn = False,mnns=None,devices=None,replace=False,scale=False,spatial=False):
    '''
    To get the alignments as a matrix showing the possibility of their alignment and the unaligned pairs are set to zero.
    Provide either the dir of adata with data_dirs or directly provide adatas.
    
    Parameters
    ----------
    data1_dir, data2_dir : str, optional
        Paths to preprocessed AnnData files
    adata1, adata2 : AnnData, optional 
        AnnData objects of two samples
    out_dim : int, default=32
        Dimension of latent features
    dropout : float, default=0.3
        Dropout probability
    lr : float, default=1e-3
        Learning rate
    min_epochs : int, default=10
        Minimum training epochs
    k : int, default=20
        Number of neighbors for initial MNN search
    min_ppf, min_percentile : float
        Parameters for filtering alignments
    min_value : float, default=0
        Minimum alignment score threshold
    percent : int, default=80
        Percentile threshold for alignments
    default_root_dir : str
        Directory for saving logs
    max_epochs : int, default=30
        Maximum training epochs
    lamb : float, default=0.2
        Hyperparameter for score-based greedy algorithm
    ckpt_dir : str, optional
        Path to load pretrained model checkpoint
    transformed : bool, default=False
        Whether input data is pre-transformed
    transformed_datas : list, optional
        Pre-transformed input data
    use_scheduler : bool, default=True
        Whether to use learning rate scheduler
    optimizer : str in ['adam','sgd'], default='adam'
        Optimizer choice
    get_edge_probs : bool, default=False
        Return raw edge probabilities
    get_matrix : bool, default=True
        Return alignment matrix
    only_mnn : bool, default=False
        Only return MNN pairs without neural network
    mnns : list, optional
        Predefined MNN pairs
    devices : list, optional
        GPU device IDs
    replace : bool, default=False
        Allow replacement in alignment selection
    scale : bool, default=False
        Scale the input data
    spatial : bool, default=False
        Use spatial information in alignment
        
    Returns
    -------
    ndarray
        Matrix of alignment probabilities between cells in the two datasets
    '''
    if (not data1_dir is None) and (not data2_dir is None):
        data1 = sc.read(data1_dir)
        data2 = sc.read(data2_dir)
    elif (not adata1 is None) and (not adata2 is None):
        data1 = adata1.copy()
        data2 = adata2.copy()
    else:
        print('Data input is not sufficient. Please provide the dir of adata or directly provide adata')
    bias = data1.shape[0]
    in_channels = data1.shape[1]
    #check if the same genes are used
    if not (data1.var_names == data2.var_names).all():
        print('The two datasets are not using the same genes')
    sc.pp.pca(data1)
    sc.pp.pca(data2)

    for data in [data1,data2]:
        if isinstance(data.X, sp.spmatrix):
            data.X = data.X.toarray()
        
    if mnns is None:
        if not transformed:
            mnn1, mnn2 = find_mutual_nn(data1.X,data2.X,k1=k,k2=k,transformed=transformed,n_jobs=-1)
            # print('finished mnn')
        else:
            mnn1, mnn2 = find_mutual_nn(transformed_datas[0],transformed_datas[1],k1=k,k2=k,transformed=transformed,n_jobs=-1)
    else:
        mnn1, mnn2 = mnns
        if len(mnn1)==0 or len(mnn2)==0:
            print('No mnn found')
            return np.zeros((data1.shape[0],data2.shape[0]))
    
    if only_mnn:
        marriage_choices = np.zeros((data1.shape[0],data2.shape[0]))
        for i,j in zip(mnn1,mnn2):
            marriage_choices[i,j] = 1
        return marriage_choices
    if scale:
        sc.pp.scale(data1)
        sc.pp.scale(data2)
    # get latent space data
    mydatamodule = MyDataModule(adata1 = data1, adata2 = data2,mnn1=mnn1,mnn2=mnn2,spatial=spatial)
    if not spatial:
        early_stopping = EarlyStopping('ap',patience=3,mode='max',min_delta=0.01)#,stopping_threshold=0.95
        Model = MSVGAE_gcl
    else:
        early_stopping = EarlyStopping('ap',patience=3,mode='max',min_delta=0.01)#,stopping_threshold=0.95
        Model = MSVGAE_gcl_spatialGW
    trainer = Trainer(max_epochs=max_epochs,devices=devices,log_every_n_steps=1,callbacks=[early_stopping,ModelSummary(max_depth=1)],default_root_dir=default_root_dir,min_epochs=min_epochs)#,RichProgressBar()
    print('start to train')
    if ckpt_dir is None:
        # model = VGAE_gcl(out_channels=out_channels,dropout=dropout,lr=lr,use_scheduler=use_scheduler,optimizer=optimizer)
        model = Model(in_channels=in_channels ,dropout=dropout,lr=lr,use_scheduler=use_scheduler,optimizer=optimizer,out_dim=out_dim,version='simple')
        start_time = time.time()
        while True:
            try :
                trainer.fit(model=model,datamodule=mydatamodule)
            except RuntimeError as e:
                    if 'CUDA out of memory' in str(e):
                        print("CUDA out of memory. Attempting to reduce batch size and retry...")
                        torch.cuda.empty_cache()  # Clear cache
                        # Optionally, reduce batch size or implement other strategies here
                        # For example, you could implement a loop to reduce the batch size
                        # and retry the current batch processing
                        time.sleep(5)  # Sleep for a few seconds
                        continue  # Skip to the next batch
                    else:
                        raise e  # Raise other errors
            break
        end_time = time.time()
        run_time = end_time - start_time
        print('Model Training Time:',run_time,'Seconds')
    else:
        # model = VGAE_gcl.load_from_checkpoint(ckpt_dir,in_channels=in_channels , out_channels=out_channels,dropout=dropout,lr=lr,use_scheduler=use_scheduler)
        model = Model.load_from_checkpoint(ckpt_dir,in_channels=in_channels ,dropout=dropout,lr=lr,use_scheduler=use_scheduler,optimizer=optimizer,out_dim=out_dim,version='simple')
        
    latent = trainer.predict(model=model,datamodule=mydatamodule)[0]
    
    # Explicit cleanup - ensure memory is released
    if 'model' in locals():
        # Move model to CPU first
        if hasattr(model, 'cuda') and next(model.parameters()).is_cuda:
            model = model.cpu()
        # Delete model
        del model
    
    if 'mydatamodule' in locals():
        # Clean up data module
        if hasattr(mydatamodule, 'train_dataset') and mydatamodule.train_dataset is not None:
            del mydatamodule.train_dataset
        if hasattr(mydatamodule, 'val_dataset') and mydatamodule.val_dataset is not None:
            del mydatamodule.val_dataset
        del mydatamodule
    
    if 'trainer' in locals():
        # Clean up trainer
        del trainer
    
    # Force garbage collection
    gc.collect()
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    if get_latent:
        return latent
    # show edge_prob
    if get_edge_probs:
        likelyhood = torch.matmul(latent[:bias], latent[bias:].T).sigmoid()
        likelyhood = np.array(likelyhood.detach().cpu()) # [data1.shape[0],data2.shape[0]]
        likelyhood[likelyhood < min_value] = 0
        return likelyhood
    # make alignment through score-based greedy algorithm
    if get_matrix:
        alignments_matrix = make_alignments(latent=latent,mnn1=mnn1,mnn2=mnn2,bias=bias,lamb=lamb,min_value=min_value,replace=replace)
        print(f'R:{data1.shape[0]} D:{data2.shape[0]}')
        return alignments_matrix

def find_mutual_nn_new(data1, data2, k1, k2, transformed_datas=None,n_jobs=-1,ckpt_dir = None,only_mnn=False,devices=[0]):
    '''
    Replacement for mnnpy
    '''
    # the input is cosine normalized and is way too small for model training
    adata1 = AnnData(X=data1) #*20
    adata2 = AnnData(X=data2) #*20
    sc.pp.scale(adata1)
    sc.pp.scale(adata2)
    alignments_matrix = get_alignments(adata1=adata1,adata2=adata2,k=k1,transformed = True,transformed_datas=transformed_datas,use_scheduler = True,ckpt_dir = ckpt_dir,lr=1e-3,default_root_dir='./Logs/mnnpy',min_percentile=85,percent=80,min_value=0.8,lamb=0.1,only_mnn=only_mnn,devices=devices)
    mutual_1 , mutual_2 = alignments_matrix.nonzero()
    # mutual_1,mutual_2 = find_mutual_nn(adata1.X,adata2.X,k1=k1,k2=k2,transformed=True,n_jobs=-1)
    return mutual_1.tolist(), mutual_2.tolist()

def get_match_scanorama(data1, data2,transformed_datas=None,ckpt_dir = None,only_mnn=False,matches=None,devices=[1]):
    '''
    replacement for scanorama
    '''
    mnn1 = []
    mnn2 = []
    for a, b in matches:
        mnn1.append(a)
        mnn2.append(b) 
    adata1 = AnnData(X=data1.toarray().astype(np.float32))#.astype(np.float32)
    adata2 = AnnData(X=data2.toarray().astype(np.float32))
    alignments_matrix = get_alignments(adata1=adata1,adata2=adata2,transformed = False,ckpt_dir = ckpt_dir,lr=1e-3,min_percentile=0,min_value=0.8,lamb=0.2,k=20,default_root_dir='./Logs/scanorama/',only_mnn=only_mnn,mnns=[mnn1,mnn2],devices=devices,scale=True)
    mutual = set()
    for i in range(alignments_matrix.shape[0]):
        for j in range(alignments_matrix.shape[1]):
            if alignments_matrix[i,j]>0:
                mutual.add((i,j))
    return mutual

def mnn_tnn(ds1, ds2, names1, names2, knn = 20,lr=1e-3,default_root_dir='./Logs/tnn_supervised/',min_ppf=0.85,min_percentile=95, min_value=0.8,percent=50,lamb = 0.3,transformed=False,transformed_datas=None,ckpt_dir=None,optimizer:Literal['adam','sgd'] = 'adam',only_mnn=False,match=None,devices=[1],scale=False):
    '''
    replacement for tnn(insct)
    '''
    if not match is None:
        mnn1 = []
        mnn2 = []
        for a, b in match:
            mnn1.append(a)
            mnn2.append(b) 
        adata1 = AnnData(X=ds1.astype(np.float32))
        adata2 = AnnData(X=ds2.astype(np.float32))
        alignments_matrix = get_alignments(adata1=adata1,adata2=adata2,k=knn,transformed = transformed,transformed_datas= transformed_datas,ckpt_dir = ckpt_dir,lr=lr,    default_root_dir=default_root_dir,min_ppf=min_ppf,min_percentile=min_percentile,min_value=min_value,percent=percent,lamb=lamb,optimizer=optimizer,only_mnn=only_mnn,mnns=[mnn1,mnn2],scale=scale,devices=devices)
    else:
        adata1 = AnnData(X=ds1.astype(np.float32))
        adata2 = AnnData(X=ds2.astype(np.float32))
        alignments_matrix = get_alignments(adata1=adata1,adata2=adata2,k=knn,transformed = transformed,transformed_datas= transformed_datas,ckpt_dir = ckpt_dir,lr=lr,    default_root_dir=default_root_dir,min_ppf=min_ppf,min_percentile=min_percentile,min_value=min_value,percent=percent,lamb=lamb,optimizer=optimizer,only_mnn=only_mnn,scale=scale,devices=devices)
    mutual = set()
    for i in range(alignments_matrix.shape[0]):
        for j in range(alignments_matrix.shape[1]):
            if alignments_matrix[i,j]>0:
                mutual.add((names1[i],names2[j]))
    return mutual

def mnn_scDML(ds1, ds2, names1, names2, knn=20,match=None,ckpt_dir = None,only_mnn=False,devices=[1]):
    #flag: in->knn, out->mnn
    # if(flag=="in"):
    #     if approx:
    #         if approx_method=="hnswlib":
    #             #hnswlib
    #             match1 = nn_approx(ds1, ds2, names1, names2, knn=knn,return_distance=return_distance,metric=metric,flag=flag)  #     save_on_disk = save_on_disk)
    #             # Find nearest neighbors in second direction.
    #             match2 = nn_approx(ds2, ds1, names2, names1, knn=knn,return_distance=return_distance,metric=metric,flag=flag)  # ,     save_on_disk = save_on_disk)
    #         else:
    #             #annoy
    #             match1 = nn_annoy(ds1, ds2, names1, names2, knn=knn,save=save,return_distance=return_distance,metric=metric,    flag=flag)  # save_on_disk = save_on_disk)
    #             # Find nearest neighbors in second direction.
    #             match2 = nn_annoy(ds2, ds1, names2, names1, knn=knn,save=save,return_distance=return_distance,metric=metric,    flag=flag)  # , save_on_disk = save_on_disk)
    
    #     else:
    #         match1 = nn(ds1, ds2, names1, names2, knn=knn, return_distance=return_distance,metric=metric,flag=flag)
    #         match2 = nn(ds2, ds1, names2, names1, knn=knn, return_distance=return_distance,metric=metric,flag=flag)
    # # Compute mutual nearest neighbors.
    
    #     if not return_distance:
    #         # mutal are set
    #         mutual = match1 | set([(b, a) for a, b in match1])
    #         return mutual
    #     else:
    #         # mutal are set
    #         mutual = set([(a, b) for a, b in match1.keys()]) | set([(b, a) for a, b in match2.keys()])
    #         #distance list of numpy array
    #         distances = []
    #         for element_i in mutual:
    #             distances.append(match1[element_i])  # distance is sys so match1[element_i]=match2[element_2]
    #         return mutual, distances
    # else:
        
    mutual = mnn_tnn(ds1, ds2, names1, names2, knn = knn,transformed=False,ckpt_dir=ckpt_dir,default_root_dir='./Logs/scDML',min_percentile=0,percent=0,min_value=0.9,lamb=0.3,lr=1e-3,only_mnn=only_mnn,match=match,devices=devices)
    # change mnn pair to symmetric
    mutual = mutual | set([(b,a) for (a,b) in mutual])
    ####################################################
    return mutual

def mnn(ds1, ds2, names1, names2, knn = 20, save_on_disk = True, approx = True):
    # Find nearest neighbors in first direction.
    if approx:
        match1 = nn_approx(ds1, ds2, names1, names2, knn=knn)#, save_on_disk = save_on_disk)
        # Find nearest neighbors in second direction.
        match2 = nn_approx(ds2, ds1, names2, names1, knn=knn)#, save_on_disk = save_on_disk)
    else:
        match1 = nn(ds1, ds2, names1, names2, knn=knn)
        match2 = nn(ds2, ds1, names2, names1, knn=knn)
    # Compute mutual nearest neighbors.
    mutual = match1 & set([ (b, a) for a, b in match2 ])

    return mutual

def mnn_tnn_spatial(ds1, ds2, spatial1,spatial2, names1, names2, knn = 20,lr=1e-3,default_root_dir='./Logs/tnn_supervised/',min_ppf=0.85,min_percentile=95, min_value=0.9,percent=50,lamb = 0.3,transformed=False,transformed_datas=None,ckpt_dir=None,optimizer:Literal['adam','sgd'] = 'adam',only_mnn=False,match=None,scale=False,devices=[0]):
    '''
    replacement for tnn(insct)
    '''
    if not match is None:
        mnn1 = []
        mnn2 = []
        for a, b in match:
            mnn1.append(a)
            mnn2.append(b) 
        adata1 = AnnData(X=ds1.astype(np.float32))
        adata2 = AnnData(X=ds2.astype(np.float32))
        adata1.obsm['spatial'] = spatial1
        adata2.obsm['spatial'] = spatial2
        alignments_matrix = get_alignments(adata1=adata1,adata2=adata2,k=knn,transformed = transformed,transformed_datas= transformed_datas,ckpt_dir = ckpt_dir,lr=lr,    default_root_dir=default_root_dir,min_ppf=min_ppf,min_percentile=min_percentile,min_value=min_value,percent=percent,lamb=lamb,optimizer=optimizer,only_mnn=only_mnn,mnns=[mnn1,mnn2],scale=scale,devices=devices,spatial=True)
    else:
        adata1 = AnnData(X=ds1.astype(np.float32))
        adata2 = AnnData(X=ds2.astype(np.float32))
        adata1.obsm['spatial'] = spatial1
        adata2.obsm['spatial'] = spatial2
        alignments_matrix = get_alignments(adata1=adata1,adata2=adata2,k=knn,transformed = transformed,transformed_datas= transformed_datas,ckpt_dir = ckpt_dir,lr=lr,    default_root_dir=default_root_dir,min_ppf=min_ppf,min_percentile=min_percentile,min_value=min_value,percent=percent,lamb=lamb,optimizer=optimizer,only_mnn=only_mnn,scale=scale,devices=devices,spatial=True)
    mutual = set()
    for i in range(alignments_matrix.shape[0]):
        for j in range(alignments_matrix.shape[1]):
            if alignments_matrix[i,j]>0:
                mutual.add((names1[i],names2[j]))
    return mutual

def mod_seurat_anchors(anchors_ori="temp/anchors.csv",adata1='temp/adata1.h5ad',adata2='temp/adata2.h5ad',min_value=0.8,lamb=0.3,devices=[2],lr=1e-3,replace=True,default_root_dir='./Logs/SeuratMod'):
    """
    change anchors matrix for Seurat-based anchors
    
    """
    anchors_ori = pd.read_csv(anchors_ori)
    mnn1 = (anchors_ori['cell1']-1).to_list()
    mnn2 = (anchors_ori['cell2']-1).to_list()
    if isinstance(adata1,str):
        adata1 = sc.read_h5ad(adata1)
    if isinstance(adata2,str):
        adata2 = sc.read_h5ad(adata2)
    alignments_matrix = get_alignments(adata1=adata1,adata2=adata2,mnns=[mnn1,mnn2],min_value=min_value,lamb=lamb,devices=devices,lr=lr,replace=replace,default_root_dir=default_root_dir)
    mutual_1 , mutual_2 = alignments_matrix.nonzero()
    score = compute_anchor_score(adata1,adata2,mutual_1,mutual_2)
    anchors_mod = pd.DataFrame({'cell1':(mutual_1+1).tolist(),'cell2':(mutual_2+1).tolist(),'score':score})
    return anchors_mod

def two_stage_spatial_imputation(adata_sn, adata_st, mnn1=None, mnn2=None, 
                                hidden_channels=128, num_layers=3, layer_type='ClusterGCN',
                                stage1_epochs=1000, similarity_weight=0.5,dropout=0., 
                                max_epochs=-1, lr=1e-3, alignment_lr=1e-3,
                                devices=[0], alignment_devices=None, k=20,default_root_dir='./logs/two_stage_spatial_imputation',
                                stage1_checkpoint=None,alignment_update_freq_delta=50,stage2_patience=20, stage2_min_delta=5e-4,
                                stage1_patience=10, stage1_min_delta=5e-4):
    """
    Two-stage spatial transcriptomics imputation with similarity preservation
    
    Parameters
    ----------
    adata_sn : AnnData
        Single-nuclei RNA-seq data (reference)
    adata_st : AnnData  
        Spatial transcriptomics data (to be imputed)
    mnn1, mnn2 : list, optional
        Pre-computed mutual nearest neighbors
    hidden_channels : int, default 512
        Hidden layer size
    num_layers : int, default 3
        Number of GNN layers
    layer_type : str, default 'GAT'
        Type of GNN layer
    stage1_epochs : int, default 50
        Number of epochs for stage 1 (imputation only)
    similarity_weight : float, default 0.5
        Weight for similarity preservation loss in stage 2
    max_epochs : int, default 100
        Maximum training epochs
    lr : float, default 1e-3
        Learning rate for main model
    alignment_lr : float, default 1e-4
        Learning rate for alignment computation
    devices : list, default [0]
        GPU devices to use for main model training
    alignment_devices : list, optional
        GPU devices to use for alignment computation. If None, uses same as devices
    k : int, default 20
        Number of neighbors for graph construction
    stage1_checkpoint : str, optional
        Path to saved stage 1 checkpoint file. If provided, will load this 
        checkpoint and skip stage 1 training, going directly to stage 2
    stage1_patience : int, default 10
        Early stopping patience for stage 1
    stage1_min_delta : float, default 1e-4
        Minimum change for early stopping in stage 1
    
    Returns
    -------
    trainer : pytorch_lightning.Trainer
        Trained model
    model : TwoStageGNNImputer
        Trained imputation model
    data_module : TwoStageDataModule
        Data module with processed data
    """
    
    # Set alignment devices to main devices if not specified
    if alignment_devices is None:
        alignment_devices = devices
    
    # Prepare data
    data_module = TwoStageDataModule(
        adata_sn=adata_sn,
        adata_st=adata_st, 
        k=k,
        mnn1=mnn1,
        mnn2=mnn2
    )
    
    # Initialize model
    if stage1_checkpoint is not None:
        # Load pre-trained stage 1 model
        print(f"Loading stage 1 checkpoint from: {stage1_checkpoint}")
        model = TwoStageGNNImputer.load_from_checkpoint(
            stage1_checkpoint,
            num_features=data_module.x.shape[1],
            n_matching_genes=data_module.n_matching_genes,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            layer_type=layer_type,
            lr=lr,
            dropout=dropout,
            stage1_epochs=0,  # Skip stage 1 since we're loading a checkpoint
            similarity_weight=similarity_weight,
            alignment_lr=alignment_lr,
            alignment_devices=alignment_devices,
            alignment_update_freq_delta=alignment_update_freq_delta,
            stage2_patience=stage2_patience, 
            stage2_min_delta=stage2_min_delta,
            stage1_patience=stage1_patience,
            stage1_min_delta=stage1_min_delta
        )
        # Mark stage 1 as complete
        model.stage1_complete = True
        model.current_epoch_stage = stage1_epochs
        print("Stage 1 checkpoint loaded. Will proceed directly to stage 2.")
        
        # Setup trainer for stage 2 only
        trainer = Trainer(
            max_epochs=max_epochs,
            devices=devices,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            log_every_n_steps=1,
            enable_checkpointing=True,
            default_root_dir=default_root_dir,
            callbacks=[
                ModelCheckpoint(
                    monitor='stage2_loss',
                    save_top_k=1,
                    mode='min'
                )
            ]
        )
    else:
        model = TwoStageGNNImputer(
            num_features=data_module.x.shape[1],
            n_matching_genes=data_module.n_matching_genes,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            layer_type=layer_type,
            lr=lr,
            dropout=dropout,
            stage1_epochs=stage1_epochs,
            similarity_weight=similarity_weight,
            alignment_lr=alignment_lr,
            alignment_devices=alignment_devices,
            alignment_update_freq_delta=alignment_update_freq_delta,
            stage2_patience=stage2_patience, 
            stage2_min_delta=stage2_min_delta,
            stage1_patience=stage1_patience,
            stage1_min_delta=stage1_min_delta
        )
        
        # Setup trainer for both stages with dynamic early stopping
        trainer = Trainer(
            max_epochs=max_epochs,
            devices=devices,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            log_every_n_steps=1,
            enable_checkpointing=True,
            default_root_dir=default_root_dir,
            callbacks=[
                ModelCheckpoint(
                    monitor='stage1_loss',
                    save_top_k=1,
                    mode='min',
                    filename='stage1-{epoch:02d}-{stage1_loss:.4f}'
                ),
                ModelCheckpoint(
                    monitor='stage2_loss',
                    save_top_k=1,
                    mode='min',
                    filename='stage2-{epoch:02d}-{stage2_loss:.4f}'
                )
            ]
        )
    
    # Train model
    trainer.fit(model, data_module)
    
    return trainer, model, data_module