## APIs

scGALA offers a comprehensive set of functions for various single-cell data integration tasks. Below are the key APIs organized by their purpose.

### Core Alignment Functions

#### `get_alignments(data1_dir=None, data2_dir=None, adata1=None, adata2=None, out_dim=32, ...)`

The main function for cell alignment between two datasets.

**Key Parameters:**
- `adata1`, `adata2`: AnnData objects containing the datasets to align
- `out_dim`: Dimension of latent features (default: 32)
- `k`: Number of neighbors for initial MNN search (default: 20)
- `min_value`: Minimum alignment score threshold (default: 0.9)
- `lamb`: Hyperparameter for score-based alignment (default: 0.3)
- `spatial`: Whether to use spatial information in alignment (default: False)
- `replace`: bool, default=False (Include the initial anchors).
        Whether to not include the initial anchors in the final alignments
- `masking_ratio` : float, default=0.3
        Ratio of masked edges during training
- `inter_edge_mask_weight` : float, default=0.5 (uniform masking).
        Weight for masking inter-dataset edges during model training.
        Higher values mean more inter-dataset edges will be removed during augmentation.

**Returns:** Matrix of alignment probabilities between cells in the two datasets

#### `find_mutual_nn_new(data1, data2, k1, k2, ...)`

Enhanced mutual nearest neighbors finding with graph learning.

**Key Parameters:**
- `data1`, `data2`: Input datasets
- `k1`, `k2`: Number of neighbors to consider in each dataset

**Returns:** Lists of mutual indices between datasets

### Seurat Integration Utilities

#### `mod_seurat_anchors(anchors_ori, adata1, adata2, min_value=0.8, lamb=0.3, ...)`

Enhance Seurat anchors using scGALA's graph-based alignment.

**Key Parameters:**
- `anchors_ori`: Path to CSV file with original anchors
- `adata1`, `adata2`: Paths to or AnnData objects for the datasets
- `min_value`: Minimum alignment score threshold (default: 0.8)
- `lamb`: Hyperparameter for anchor refinement (default: 0.3)

**Returns:** Enhanced alignment matrix

#### `compute_anchor_score(adata1, adata2, mnn1, mnn2)`

Calculate anchor scores for pairs of cells, useful for downstream integration tasks.

**Key Parameters:**
- `adata1`, `adata2`: AnnData objects for the datasets
- `mnn1`, `mnn2`: Lists of indices representing aligned cell pairs

**Returns:** Array of anchor scores

### Batch Correction Integration (the modified function for existing methods)

#### `mnn_tnn(ds1, ds2, names1, names2, knn=20, ...)`

Replace TNN (INSCT) alignment with scGALA-enhanced alignment.

**Key Parameters:**
- `ds1`, `ds2`: Input datasets
- `names1`, `names2`: Cell names for each dataset
- `knn`: Number of neighbors for alignment (default: 20)
- `min_value`: Minimum alignment score threshold (default: 0.8)

**Returns:** Aligned indices between datasets

#### `get_match_scanorama(data1, data2, ...)`

Enhanced alignment for Scanorama integration method.

**Key Parameters:**
- `data1`, `data2`: Input datasets
- `matches`: Optional pre-computed matches

**Returns:** Aligned indices between datasets

#### `mnn_scDML(ds1, ds2, names1, names2, knn=20, ...)`

Enhanced alignment for scDML batch correction method.

**Key Parameters:**
- `ds1`, `ds2`: Input datasets
- `names1`, `names2`: Cell names for each dataset
- `knn`: Number of neighbors (default: 20)

**Returns:** Aligned indices between datasets

### Spatial Transcriptomics Tools

#### `mnn_tnn_spatial(ds1, ds2, spatial1, spatial2, names1, names2, ...)`

Spatial-aware version of mnn_tnn that incorporates spatial coordinates.

**Key Parameters:**
- `ds1`, `ds2`: Input expression datasets
- `spatial1`, `spatial2`: Spatial coordinate information
- `names1`, `names2`: Cell names for each dataset
- `min_value`: Minimum alignment score threshold (default: 0.9)

**Returns:** Aligned indices between spatial datasets

#### `GNNImputer` (class)

Neural network model for imputing gene expression in spatial data.

**Key Parameters in constructor:**
- `num_features`: Number of input features
- `n_matching_genes`: Number of genes shared between datasets
- `hidden_channels`: Size of hidden layers
- `num_layers`: Number of GNN layers (default: 3)
- `layer_type`: Type of GNN layer to use (default: 'GAT')

**Usage:** Used through `MyDataModule_OneStage` in the spatial imputation workflow

### Data Handling Utilities

#### `split_data_unevenly(adata, train_ratio=0.7, group_key='cell_type')`

Split datasets with controlled imbalance for robust testing.

**Key Parameters:**
- `adata`: Input AnnData object
- `train_ratio`: Overall ratio for training set (default: 0.7)
- `group_key`: Column in adata.obs for splitting (default: 'cell_type')

**Returns:** Two AnnData objects (train and test)

#### `simulate_batch_effect(data, batch_effect_strength=0.3, noise_strength=0.3)`

Add synthetic batch effects for benchmarking integration methods.

**Key Parameters:**
- `data`: Input data matrix
- `batch_effect_strength`: Strength of batch effect (default: 0.3)
- `noise_strength`: Strength of random noise (default: 0.3)

**Returns:** Data with simulated batch effects