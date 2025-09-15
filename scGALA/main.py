import warnings
import scanpy as sc
from .model import MSVGAE_gcl,MSVGAE_gcl_spatialGW
from .data import MyDataModule
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping,ModelSummary
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

def get_alignments(data1_dir=None, data2_dir=None,adata1=None,adata2=None, out_dim:int = 32, dropout:float = 0.3, lr:float = 1e-3,min_epochs:int = 10, k:int =20, min_value=0.9, default_root_dir=None,max_epochs:int = 30,lamb = 0.3,ckpt_dir = None, transformed = False, transformed_datas=None, use_scheduler:bool = True,optimizer:Literal['adam','sgd'] = 'adam',get_latent:bool = False, get_edge_probs:bool = False, get_matrix:bool = True,only_mnn = False,mnns=None,devices=None,replace=False,scale=False,spatial=False, masking_ratio=0.3,inter_edge_mask_weight:float = 0.5):
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
    replace : bool, default=False (Include the initial anchors).
        Whether to not include the initial anchors in the final alignments
    scale : bool, default=False
        Scale the input data
    spatial : bool, default=False
        Use spatial information in alignment
    masking_ratio : float, default=0.3
        Ratio of masked edges during training
    inter_edge_mask_weight : float, default=0.5
        Weight for masking inter-dataset edges during model training.
        Higher values mean more inter-dataset edges will be removed during augmentation.
        
    Returns
    -------
    ndarray
        Matrix of alignment probabilities between cells in the two datasets
    '''
    if (not data1_dir is None) and (not data2_dir is None):
        data1 = sc.read(data1_dir)
        data2 = sc.read(data2_dir)
    elif (not adata1 is None) and (not adata2 is None):
        data1 = adata1
        data2 = adata2
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
        model = Model(in_channels=in_channels ,dropout=dropout,lr=lr,masking_ratio=masking_ratio,use_scheduler=use_scheduler,optimizer=optimizer,out_dim=out_dim,version='simple',inter_edge_mask_weight=inter_edge_mask_weight)
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
        model = Model.load_from_checkpoint(ckpt_dir,in_channels=in_channels ,dropout=dropout,lr=lr,masking_ratio=masking_ratio,use_scheduler=use_scheduler,optimizer=optimizer,out_dim=out_dim,version='simple',inter_edge_mask_weight=inter_edge_mask_weight)
        
    latent = trainer.predict(model=model,datamodule=mydatamodule)[0]
    
    # clean up
    torch.cuda.empty_cache()
    model = None
    mydatamodule = None
    
    if get_latent:
        return latent
    # show edge_prob
    if get_edge_probs:
        likelyhood = torch.matmul(latent[:bias], latent[bias:].T).sigmoid()
        likelyhood = np.array(likelyhood.detach().cpu()) # [data1.shape[0],data2.shape[0]]
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