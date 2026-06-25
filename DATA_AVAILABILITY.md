# Data Availability

This release includes the Pancreas example dataset and the gene-set files required by the example workflow:

- `data/datasets/Pancreas/data.h5ad`
- `data/gene_sets/human_function3/Hu_GO_bp.gmt`
- `data/gene_sets/human_function3/gene_sets.gmt`

The gene-set files are derived from MSigDB. Users should follow the original MSigDB terms of use when using or redistributing these files.

The Pancreas example data are derived from GEO accession [GSE132188](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE132188), with corresponding SRA accession [SRP200419](https://www.ncbi.nlm.nih.gov/sra/?term=SRP200419) and BioProject accession [PRJNA546282](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA546282). Users should follow the terms associated with the original Pancreas dataset when reusing or redistributing the data.

The repository license applies to the source code in this release. It does not override the original licenses or terms of use for the included datasets or gene-set resources.

Large data files are configured for Git LFS through `.gitattributes`.
