import os
import gc
import h5py
import hnswlib
import pandas as pd
import numpy as np
import scanpy as sc
import scipy as sp

class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def empty_safe(fn, dtype):
    def _fn(x):
        if x.size:
            return fn(x)
        return x.astype(dtype)
    return _fn


decode = empty_safe(np.vectorize(lambda _x: _x.decode("utf-8")), str)
encode = empty_safe(np.vectorize(lambda _x: str(_x).encode("utf-8")), "S")
upper = empty_safe(np.vectorize(lambda x: str(x).upper()), str)
lower = empty_safe(np.vectorize(lambda x: str(x).lower()), str)
tostr = empty_safe(np.vectorize(str), str)


def read_clean(data):
    assert isinstance(data, np.ndarray)
    if data.dtype.type is np.bytes_:
        data = decode(data)
    if data.size == 1:
        data = data.flat[0]
    return data


def dict_from_group(group):
    assert isinstance(group, h5py.Group)
    d = dotdict()
    for key in group:
        if isinstance(group[key], h5py.Group):
            value = dict_from_group(group[key])
        else:
            value = read_clean(group[key][...])
        d[key] = value
    return d


def read_data(filename, sparsify = False, skip_exprs = False):
    with h5py.File(filename, "r") as f:
        obs = pd.DataFrame(dict_from_group(f["obs"]), index = decode(f["obs_names"][...]))
        var = pd.DataFrame(dict_from_group(f["var"]), index = decode(f["var_names"][...]))
        uns = dict_from_group(f["uns"])

        if not skip_exprs:
            exprs_handle = f["exprs"]
            if isinstance(exprs_handle, h5py.Group):
                mat = sp.sparse.csr_matrix((exprs_handle["data"][...], 
                                            exprs_handle["indices"][...],
                                            exprs_handle["indptr"][...]), 
                                            shape = exprs_handle["shape"][...])
            else:
                mat = exprs_handle[...].astype(np.float32)
                if sparsify:
                    mat = sp.sparse.csr_matrix(mat)
        else:
            mat = sp.sparse.csr_matrix((obs.shape[0], var.shape[0]))

    return mat, obs, var, uns


def prepro(filename):
    data_path = os.path.join(filename, "data.h5")
    mat, obs, var, uns = read_data(data_path, sparsify=False, skip_exprs=False)

    if isinstance(mat, np.ndarray):
        X = np.array(mat)
    else:
        X = np.array(mat.toarray())

    cell_name = np.array(obs["cell_type1"])
    cell_type, cell_label = np.unique(cell_name, return_inverse=True)

    return X, cell_label, cell_name

def prepare(filename):
    data_path = os.path.join(filename, "data.h5")
    mat, obs, var, uns = read_data(data_path, sparsify=False, skip_exprs=False)

    if isinstance(mat, np.ndarray):
        X = np.array(mat)
    else:
        X = np.array(mat.toarray())

    cell_name = np.array(obs["cell_type1"])
    cell_type, cell_label = np.unique(cell_name, return_inverse=True)

    return X, cell_label, cell_name, var.index.to_list()


def normalize(adata, 
              copy=True,
              flavor=None, 
              highly_genes=None, 
              filter_min_counts=True, 
              normalize_input=True, 
              logtrans_input=True,
              scale_input=False):
    if isinstance(adata, sc.AnnData):
        if copy:
            adata = adata.copy()
    else:
        raise NotImplementedError

    if filter_min_counts:
        sc.pp.filter_genes(adata, min_cells=3)

    if normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata

    if flavor == 'seurat_v3':
        print("seurat_v3")
        sc.pp.highly_variable_genes(adata, flavor=flavor, n_top_genes = highly_genes)

    if normalize_input:
        sc.pp.normalize_total(adata, target_sum=1e4)

    if logtrans_input:
        sc.pp.log1p(adata)

    if flavor is None:
        if highly_genes is not None:
            print("routine hvg")
            sc.pp.highly_variable_genes(adata, n_top_genes=highly_genes)
        else:
            sc.pp.highly_variable_genes(adata)
    
    adata_hvg = adata[:, adata.var.highly_variable].copy()

    if scale_input:
        sc.pp.scale(adata_hvg)

    return adata, adata_hvg

def cal_nn(x, k=30, max_element=95536):
    p = hnswlib.Index(space='cosine', dim=x.shape[1])
    p.init_index(max_elements = max_element, ef_construction = 300, M = 100)
    p.set_num_threads(10)
    p.set_ef(600)
    p.add_items(x)

    neighbors, distance = p.knn_query(x, k = k)
    neighbors = neighbors[:, 1:]
    distance = distance[:, 1:]

    return neighbors, distance

def gen_tf_gene_table(genes, tf_list, dTD):
    gene_names = [g.upper() for g in genes]
    TF_names = [g.upper() for g in tf_list]
    tf_gene_table = dict.fromkeys(tf_list)

    for i, tf in enumerate(tf_list):
        tf_gene_table[tf] = np.zeros(len(gene_names))
        _genes = dTD[tf]

        _existed_targets = list(set(_genes).intersection(gene_names))
        _idx_targets = map(lambda x: gene_names.index(x), _existed_targets)

        for _g in _idx_targets:
            tf_gene_table[tf][_g] = 1

    del gene_names
    del TF_names
    del _genes
    del _existed_targets
    del _idx_targets

    gc.collect()

    return tf_gene_table

 
def getGeneSetMatrix(_name, genes_upper, gene_sets_path):
    if _name[-3:] == 'gmt' or _name[-3:] == 'txt':
        print(f"GMT or TXT file {_name} loading ... ")
        filename = _name
        filepath = os.path.join(gene_sets_path, f"{filename}")
        print(filepath)

        with open(filepath) as genesets:
            pathway2gene = {line.strip().split("\t")[0]: line.strip().split("\t")[2:]
                            for line in genesets.readlines()}

        print(len(pathway2gene))

        gs = []
        for k, v in pathway2gene.items():
            gs += v

        pathway_list = pathway2gene.keys()
        pathway_gene_table = gen_tf_gene_table(genes_upper, pathway_list, pathway2gene)
        gene_set_matrix = np.array(list(pathway_gene_table.values()))
        keys = pathway_gene_table.keys()

        del pathway2gene
        del gs
        del pathway_list

    else:
        gene_set_matrix = None

    return gene_set_matrix, keys, pathway_gene_table



