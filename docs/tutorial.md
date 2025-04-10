## Tutorials
### Integrate into existing methods (Harmonization Pipeline and Universal Booster).
scGALA is designed to easily integrate into existing methods that employ cell alignments. The integration can be done in two modes: Module Replacement or External Reference, depending on the working strategy of the target method.

For methods under develop or run in an end-to-end way, then Module Replacement is the strategy to choose. Identify the Cell Alignment module (key words to look for: MNN, Alignment, CCA, Anchor, Correspondence) and replace it with scGALA as in the Usage. 

Tutorial with INSCT as example: {doc}`Module Replacement Tutorial Based On INSCT (Batch Correction) </tutorials/Batch_Correction_Comparison_INSCT_Supervised>`. We presented the comparison experiment between scGALA-enhanced INSCT Supervised and original INSCT Supervised. To facilitate the evaluation, we use [scIB](https://github.com/theislab/scib) to compute core metrics of batch correction. 

For methods with clear procedure steps, we recommend the External Reference strategy, as this needs least efforts. In this mode, we don't replace the alignment module, instead, we enhance the intermediate cell alignment results given by the original method.

Tutorial with Seurat as example: {doc}`External Reference Tutorial Based On Seurat (Label Transfer) </tutorials/Label_Transfer_Comparison_Seurat>`.  We presented the comparison experiment between scGALA-enhanced Seurat and original Seurat. We proved APIs to efficiently enhance seurat-based anchors and compute anchor scores required by Seurat.

More tutorials are provided to demonstrate {doc}`Multiomcs Integration based on scGALA-enhanced Seurat </tutorials/Multiomics_Integration_Seurat>` and {doc}`Spatial Alignment based on scGALA-enhanced STAligner </tutorials/Spatial_Alignment_STAligner>`.

### Advanced Multiomics Functionalities
#### Multiplet-omics Integration
scGALA introduces a multiplet-omics integration strategy that bridges disjoint doublet datasets, such as RNA-ATAC and RNA-ADT, to computationally construct a triplet-omics dataset (RNA-ATAC-ADT), thus bypassing the need for specialized triple-modal sequencing protocols while maintaining coherence across modalities.

Tutorial: {doc}`Multiplet-omics Integration with scGALA </tutorials/Multiplet-omics_Integration>`. We demonstrated how scGALA can be used for multiplet-omics integration, specifically integrating RNA+ATAC and RNA+ADT datasets through their shared RNA modality. 
#### Cross-modality Imputation and Generation
scGALA enables cross-modality data generation through a specialized Graph Attention Network framework. This allows for predicting RNA expression profiles from chromatin accessibility (ATAC-seq) data, effectively creating multimodal profiles from single-modality measurements.

Tutorial: {doc}`Cross-modality Imputation with scGALA </tutorials/Cross-modality_Imputation>`. We demonstrate how to use scGALA to generate gene expression (RNA) profiles from ATAC-seq data using cell-cell alignments as guiding information.
#### Spatial Transcriptomics Enhancement
scGALA offers functionality to impute spatial transcriptomics data with the help of a reference scRNA dataset. This addresses a major limitation of spatial technologies, which typically measure only a few hundred genes compared to thousands in scRNA-seq.

Tutorial: {doc}`Spatial Transcriptomics Imputation with scGALA </tutorials/Spatial_Transcriptomics_Imputation>`. We show how to enhance spatially resolved transcriptomics by imputing unmeasured genes using a reference scRNA-seq dataset while preserving spatial context.