import os
import scanpy as sc
import scipy as sp
import numpy as np
import pandas as pd
from utils.tools import normalize

def prepare_npz(file_name):
    dname = file_name.split("/")[-1]
    
    X = sp.sparse.load_npz(os.path.join(file_name, "filtered_Counts.npz"))
    X = sp.sparse.csr_matrix(X)
    X = X.toarray()
    annoData = pd.read_table(os.path.join(file_name, "annoData.txt"))
    genes = pd.read_table(os.path.join(file_name, "genes.txt"), header=None).iloc[:, 0].values

    if dname == "HCF-spleen":
        Y = annoData['cellIden'].to_numpy()
        cell_type = list(map(str, Y))
    else:
        if "cellIden3" in annoData.columns.to_list():
            Y = annoData['cellIden3'].to_numpy()
        else:
            Y = annoData['cellIden'].to_numpy()

        if "cellAnno3" in annoData.columns.to_list():
            cell_type = annoData['cellAnno3'].to_list()
        elif "celltype" in annoData.columns.to_list():
            cell_type = annoData['celltype'].to_list()
        else:
            cell_type = annoData['cellAnno'].to_list()

    return X, Y, cell_type, genes

def prepare_h5ad(file_name):
    name = file_name.split("/")[-1]
    adata = sc.read_h5ad(os.path.join(file_name, "data.h5ad"))
    
    if name == "DLPFC":
        adata = adata[~adata.obs['Ground Truth'].isna()].copy()

    X = sp.sparse.csr_matrix(adata.X)
    X = X.toarray()
    
    print(adata)
    
    if name == "Pancreas":
        cell_name = np.array(adata.obs['clusters'].values)
    elif name == "PBMC3K_processed":
        cell_name = np.array(adata.obs['louvain'].values)
    elif name == "DLPFC":
        cell_name = np.array(list(map(str, adata.obs['Ground Truth'].values)))
    elif name == 'forebrain':
        cell_name = np.array(list(map(str, adata.obs['Clusters'].values)))
    else:
        cell_name = np.array(adata.obs["cell_type"])
        
    cell_type, Y = np.unique(cell_name, return_inverse=True)

    return X, Y, cell_name, adata.var_names.to_list()

def prepare_h5(file_name):
    import h5py
    data_mat = h5py.File(os.path.join(file_name, "data.h5"), "r")

    X = np.array(data_mat['X'])
    cell_name = np.array(data_mat['Y'])

    cell_type, Y = np.unique(cell_name, return_inverse=True)

    return X, Y, cell_name, None

def prepare_nested_h5(file_name):
    # X, Y, cell_type = prepro(file_name)
    X, Y, cell_type, genes = prepare(file_name)

    X = np.ceil(X).astype(np.int32)


    return X, Y, cell_type, genes

def unify_prepare(root_dir, 
                  data_type, 
                  dataset_name, 
                  num_genes, 
                  scale=False):
    file_name = os.path.join(root_dir, dataset_name)
    print(file_name)
    print(f"Current Processed Dataset is: {dataset_name}")

    if data_type == "npz":
        X, Y, cell_type, genes = prepare_npz(file_name)
    elif data_type == 'h5_nested':
        X, Y, cell_type, genes = prepare_nested_h5(file_name)
    elif data_type == "h5":
        X, Y, cell_type, genes = prepare_h5(file_name)
    elif data_type == 'h5ad':
        X, Y, cell_type, genes = prepare_h5ad(file_name)
    else:
        raise Exception("Please Input Proper Data Type!")
    
    adata = sc.AnnData(X, dtype=np.float32)
    if genes is not None:
        adata.var_names = genes
    adata.obs['Group'] = Y
    adata.obs['annotation'] = cell_type
    
    num_genes = min(num_genes, adata.X.shape[1])
    print(f"num_genes = {num_genes}")

    if dataset_name != 'PBMC3K_processed' and dataset_name != 'DLPFC':
        adata, adata_hvg = normalize(adata, 
                                    copy=True,
                                    flavor=None,
                                    highly_genes=num_genes,
                                    normalize_input=True,
                                    logtrans_input=True,
                                    scale_input=scale)
    elif dataset_name == "DLPFC":
        adata, adata_hvg = normalize(adata, 
                                    copy=True,
                                    flavor='seurat_v3',
                                    highly_genes=num_genes,
                                    normalize_input=True,
                                    logtrans_input=True,
                                    scale_input=scale)
    else:
        adata_hvg = adata.copy()
        
    return adata, adata_hvg, genes, cell_type

