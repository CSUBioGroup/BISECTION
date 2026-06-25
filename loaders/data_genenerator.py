from read_data import unify_prepare
from utils.tools import cal_nn

def data_process(root_dir, 
                 data_type, 
                 dataset_name, 
                 num_genes, 
                 k=6, 
                 max_element=95536, 
                 scale=False):
    adata_raw, adata_hvg, all_genes, cell_type = unify_prepare(root_dir, 
                                                                data_type, 
                                                                dataset_name, 
                                                                num_genes, 
                                                                scale)
     
    x_array = adata_hvg.to_df().values
    y_array = adata_hvg.obs['Group'].values

    print(f"X shape: {x_array.shape}")
    print(f"Y shape: {y_array.shape}")
    
    if k > 0:
        neighbors, _ = cal_nn(x_array, k=k, max_element=max_element)
    else:
        return x_array, y_array, cell_type, None, None
    
    if all_genes is None:
        return x_array, y_array, cell_type, neighbors, adata_hvg, None

    return x_array, y_array, cell_type, neighbors, adata_hvg, adata_hvg.var_names

    

