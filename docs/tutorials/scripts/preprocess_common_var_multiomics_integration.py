import anndata
import scanpy as sc
import sccross
from scipy.sparse import csr_matrix
import episcanpy as epi
import sys

data_dir = sys.argv[1]
rna = anndata.read_h5ad(data_dir+'/RNA.h5ad')
atac = anndata.read_h5ad(data_dir+'/ATAC.h5ad')
rna.layers["counts"] = rna.X.copy()


sccross.data.lsi(atac, n_components=100, n_iter=15)
sc.pp.neighbors(atac, use_rep="X_lsi", metric="cosine")
sc.tl.umap(atac)

atac2rna = epi.tl.geneactivity(atac,
                                gtf_file='./reference/gencode.vM30.annotation.gtf',
                                key_added='gene',
                                upstream=2000,
                                feature_type='transcript',
                                annotation='HAVANA',
                                raw=False)
atac2rna.X = csr_matrix(atac2rna.X)

common_genes = set()
adatas = [rna, atac2rna]
for i in range(len(adatas)):
    adata_i = adatas[i].var.index.values
    atac_i_1 = []
    for g in adata_i:
        g = g.split('-')[0]
        atac_i_1.append(g)
    ## atac2rna has the same value under gene name with different suffixes: Hym_32, Hym_35...
    adatas[i].var.index = atac_i_1
    adatas[i].var_names_make_unique()
    if len(common_genes) == 0:
        common_genes = set(adatas[i].var.index.values)
    else:
        common_genes &= set(adatas[i].var.index.values)
for i in range(len(adatas)):
    adatas[i] = adatas[i][:, list(common_genes)]
rna = adatas[0]
atac2rna = adatas[1]

sc.pp.highly_variable_genes(rna, n_top_genes=2000, flavor="seurat_v3")
sc.pp.normalize_total(rna)
sc.pp.log1p(rna)
sc.pp.scale(rna)
sc.tl.pca(rna, n_comps=100, svd_solver="auto")
sc.pp.neighbors(rna, metric="cosine")
sc.tl.umap(rna)

sc.pp.normalize_total(atac2rna)
sc.pp.log1p(atac2rna)
sc.pp.scale(atac2rna)

atac.uns['gene'] = atac2rna

rna.write(data_dir+'/seurat'+"/rna_preprocessed.h5ad", compression="gzip")
atac.write(data_dir+'/seurat'+"/atac_preprocessed.h5ad", compression="gzip")
atac2rna.write(data_dir+'/seurat'+"/atac2rna.h5ad", compression="gzip")