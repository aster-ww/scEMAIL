# scEMAIL
### Universal and Source-free Annotation Method for scRNA-seq Data with Novel Cell-type Perception

Current cell-type annotation tools offor single-cell RNA sequencing (scRNA-seq) data mainly utilize well-annotated source data to help identify cell types in target data. However, on account of privacy preservation, their requirements for raw source data may not always be satisfied. In this case, achieving feature alignment between source and target data explicitly is impossible. Additionally, these methods are barely able to discover the presence of novel cell types. A subjective threshold is often selected by users to detect novel cells. We propose a universal annotation framework for scRNA-seq data called scEMAIL, which automatically detects novel cell types without accessing source data during adaptation. For new celltype identification, a novel cell-type perception module is designed with three steps. First, an expert ensemble system measures uncertainty of each cell from three complementary aspects. Second, based on this measurement, bimodality tests are applied to detect the presence of new cell types. Third, once assured of their presence, an adaptive threshold via manifold mixup partitions target cells into “known” and “unknown” groups. Model adaptation is then conducted to alleviate the batch effect. We gather multi-order neighborhood messages globally and impose local affinity regularizations on “known” cells. These constraints mitigate wrong classifications of the source model via reliable self-supervised information of neighbors. scEMAIL is accurate and robust under various scenarios in both simulation and real data. It is also flexible to be applied to challenging single-cell ATAC-seq data without loss of superiority.<br>  
<br> ![](https://ars.els-cdn.com/content/image/1-s2.0-S1672022922001747-gr1.jpg)
<br> 
For more information, please refer to https://doi.org/10.1016/j.gpb.2022.12.008

## Reference
Wan, H., Chen, L., & Deng, M. (2023). scEMAIL: Universal and Source-free Annotation Method for scRNA-seq Data with Novel Cell-type Perception. Genomics, Proteomics & Bioinformatics.
<br>
