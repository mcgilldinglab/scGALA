# scGALA
scGala: Graph Link Prediction Based Cell Alignment for Comprehensive Data Integration

For detailed instructions, comprehensive documentation, and helpful tutorials, please visit: 
- https://scgala.readthedocs.io/en/latest/intro.html
## Overview
<img title="scGALA Overview" alt="Alt text" src="images/scGALAOverview.png">

## Installation

**Step 1**: Create a [conda](https://docs.anaconda.com/miniconda/install/#quick-command-line-install) environment for scGALA

```bash
conda create -n scGALA python=3.10 -y

conda activate scGALA
``` 
**Step 2**:
Install Pytorch as described in its [official documentation](https://pytorch.org/get-started/locally/). Choose the platform and accelerator (GPU/CPU) accordingly to avoid common dependency issues.

> **A note regarding Additional Libraries for required package PyG**
>
> For required additional libraries in [PyG](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html), it is hard for PyPI to determine the correct version to install. Please install the additional dependencies accordingly after install scGALA.

```bash
# Pytorch example, choose the cuda version accordingly
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
# Install scGALA
pip install scGALA
# Example for PyG additional dependencies. Please read the note and install them based on your actual hardware.
# PyG
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
``` 

## Usage:
For the core function, which is the cell alignment in scGALA, simple run:
```python
from scGALA import get_alignments

# You can get the edge probability matrix for one line
alignment_matrix = get_alignments(adata1=adata1,adata2=adata2)

# To get the anchor index for two datasets
anchor_index1, anchor_index2 = alignments_matrix.nonzero()

# The anchor cells are easy to obtain by
anchor_cell1 = adata1[anchor_index1]
anchor_cell2 = adata2[anchor_index1]
```

We also provide convenient APIs for enhancing Seurat-based anchors, imputing spatial transcriptomics, generating cross-modality data, and other useful features. Please refer to Tutorials and APIs for detailed walkthroughs.

## Example Data
All example data used in the Tutorials can be found in [Figshare](https://figshare.com/articles/dataset/Label_Transfer_Example_Data/28728617). The data used in batch correciton tutorial can be found in [Figshare](https://figshare.com/articles/dataset/Benchmarking_atlas-level_data_integration_in_single-cell_genomics_-_integration_task_datasets_Immune_and_pancreas_/12420968).

## Tutorials
### Integrate into existing methods (Harmonization Pipeline and Universal Booster).
scGALA is designed to easily integrate into existing methods that employ cell alignments. The integration can be done in two modes: Module Replacement or External Reference, depending on the working strategy of the target method.

For methods under develop or run in an end-to-end way, then Module Replacement is the strategy to choose. Identify the Cell Alignment module (key words to look for: MNN, Alignment, CCA, Anchor, Correspondence) and replace it with scGALA as in the Usage. 

Tutorial with INSCT as example: [Module Replacement Tutorial Based On INSCT (Batch Correction)](https://scgala.readthedocs.io/en/latest/tutorials/Batch_Correction_Comparison_INSCT_Supervised.html). We presented the comparison experiment between scGALA-enhanced INSCT Supervised and original INSCT Supervised. To facilitate the evaluation, we use [scIB](https://github.com/theislab/scib) to compute core metrics of batch correction. 

For methods with clear procedure steps, we recommend the External Reference strategy, as this needs least efforts. In this mode, we don't replace the alignment module, instead, we enhance the intermediate cell alignment results given by the original method.

Tutorial with Seurat as example: [External Reference Tutorial Based On Seurat (Label Transfer)](https://scgala.readthedocs.io/en/latest/tutorials/Label_Transfer_Comparison_Seurat.html).  We presented the comparison experiment between scGALA-enhanced Seurat and original Seurat. We proved APIs to efficiently enhance seurat-based anchors and compute anchor scores required by Seurat.

More tutorials are provided to demonstrate [Multiomcs Integration based on scGALA-enhanced Seurat](https://scgala.readthedocs.io/en/latest/tutorials/Multiomics_Integration_Seurat.html) and [Spatial Alignment based on scGALA-enhanced STAligner](https://scgala.readthedocs.io/en/latest/tutorials/Spatial_Alignment_STAligner.html).

### Advanced Multiomics Functionalities
#### Multiplet-omics Integration
scGALA introduces a multiplet-omics integration strategy that bridges disjoint doublet datasets, such as RNA-ATAC and RNA-ADT, to computationally construct a triplet-omics dataset (RNA-ATAC-ADT), thus bypassing the need for specialized triple-modal sequencing protocols while maintaining coherence across modalities.

Tutorial: [Multiplet-omics Integration with scGALA](https://scgala.readthedocs.io/en/latest/tutorials/Multiplet-omics_Integration.html). We demonstrated how scGALA can be used for multiplet-omics integration, specifically integrating RNA+ATAC and RNA+ADT datasets through their shared RNA modality. 
#### Cross-modality Imputation and Generation
scGALA enables cross-modality data generation through a specialized Graph Attention Network framework. This allows for predicting RNA expression profiles from chromatin accessibility (ATAC-seq) data, effectively creating multimodal profiles from single-modality measurements.

Tutorial: [Cross-modality Imputation with scGALA](https://scgala.readthedocs.io/en/latest/tutorials/Cross-modality_Imputation.html). We demonstrate how to use scGALA to generate gene expression (RNA) profiles from ATAC-seq data using cell-cell alignments as guiding information.
#### Spatial Transcriptomics Enhancement
scGALA offers functionality to impute spatial transcriptomics data with the help of a reference scRNA dataset. This addresses a major limitation of spatial technologies, which typically measure only a few hundred genes compared to thousands in scRNA-seq.

Tutorial: [Spatial Transcriptomics Imputation with scGALA](https://scgala.readthedocs.io/en/latest/tutorials/Spatial_Transcriptomics_Imputation.html). We show how to enhance spatially resolved transcriptomics by imputing unmeasured genes using a reference scRNA-seq dataset while preserving spatial context.

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
