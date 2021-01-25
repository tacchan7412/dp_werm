# Differentially Private Weighted Empirical Risk Minimization for Imbalanced Dataset

## Dependencies
- Python (3.6)
- R (3.6.1)
  - edgeR (3.26.8)

## Running Experiments
1. Download datasets
    1. [KDDCup99](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html)
    Download `kddcup.data_10_percent` and `corrected` and put them under `data/kdd99/raw`
    2. RNA-seq
        1. [TCGA](https://xenabrowser.net/datapages/?dataset=tcga_RSEM_gene_tpm&host=https://toil.xenahubs.net)
        Download `tcga_RSEM_gene_tpm` and `gencode.v23.annotation.gene.probeMap` and put them under `data/rnaseq/raw`
        2. [GTEx](https://xenabrowser.net/datapages/?dataset=gtex_RSEM_gene_tpm&host=https%3A%2F%2Ftoil.xenahubs.net&removeHub=https%3A%2F%2Fxena.treehouse.gi.ucsc.edu%3A443)
        Download `gtex_RSEM_gene_tpm` and put it under `data/rnaseq/raw`
2. Execute the following command
```
python tune.py [ loss function ] [ algorithm name ] [ dataset ] [ random seed ] [ eps ] [ ratio of positive labels (for synthetic only) ]
```
example:
```
python tune.py lr  obj_pert_ew synthetic_dif 1 0.5 0.1
```
