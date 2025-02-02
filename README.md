# scGALA
scGala: Graph Link Prediction Based Cell Alignment for Comprehensive Data Integration
## Overview
<img title="scGALA Overview" alt="Alt text" src="scGALA Overview.png">

## Installation

**Step 1**: Create a [conda](https://docs.anaconda.com/miniconda/install/#quick-command-line-install) environment for scGALA

```bash
conda create -n scGALA python=3.10

conda activate scGALA
``` 
**Step 2**: Install the dependencies.
Install Pytorch via conda install as described in its [official documentation](https://pytorch.org/get-started/locally/). Choose the platform and accelerator (GPU/CPU) accordingly to avoid common dependency issues.

Install [PyG](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html), [PyGCL](https://github.com/PyGCL/PyGCL), [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/)
```bash
# PyG
conda install pyg -c pyg
# PyGCL
pip install PyGCL
# PyTorch Lightning
conda install lightning -c conda-forge
``` 
> **A note regarding DGL for package PyGCL**
>
> Currently the DGL team maintains two versions, `dgl` for CPU support and `dgl-cu***` for CUDA support. Since `pip` treats them as different packages, it is hard for PyGCL to check for the version requirement of `dgl`. They have removed such dependency checks for `dgl` in our setup configuration and require the users to [install a proper version](https://www.dgl.ai/pages/start.html) by themselves.

**Step 3**: Clone This Repo

```bash
git clone https://github.com/mcgilldinglab/scGALA.git
```

## Usage:
For the main function, which is the cell alignment in scGALA, simple run:
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