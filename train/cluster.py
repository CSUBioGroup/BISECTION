import os
import time

import numpy as np
import prettytable as pt
import torch
from torch.utils.data import DataLoader

from evaluation import (
    embedding_cluster_visualization_sc,
    evaluate_kmeans,
    evaluate_leiden,
    run_kmeans,
    run_leiden,
)
from loaders import CellDataset


def predict(args, model, x, y, device):
    model.eval()
    val_datasets = CellDataset(x, y)
    in_features = val_datasets.data.size(1)

    print(f"Validation Dataset size: {len(val_datasets)}")
    print(f"The in_features is: {in_features}")

    val_loader = DataLoader(val_datasets,
                            batch_size=256,
                            shuffle=False,
                            num_workers=args.workers,
                            drop_last=False)

    labels_vector = []
    latent_vector = []
    for step, (x, y) in enumerate(val_loader):
        x = x.to(device)

        with torch.no_grad():
            latent = model.get_embedding(x)

        latent = latent.detach()

        latent_vector.extend(latent.cpu().detach().numpy())
        labels_vector.extend(y.numpy())

        if step % 50 == 0:
            print(f"Step [{step}/{len(val_loader)}]\t Computing features...")

    labels_vector = np.array(labels_vector)
    latent_vector = np.array(latent_vector)

    return labels_vector, latent_vector, val_loader

def get_pseudo_labels(args, 
                      model, 
                      x,
                      y,
                      cell_type,
                      device):
    Y, latent, val_loader = predict(args, model, x, y, device)
    print("### Performming Leiden clustering method on latent vector ###")
    adata_embedding, leiden_pred = run_leiden(latent_vector=latent, 
                                              resolution=args.resolution)
    
    print("### Performming KMeans clustering method on latent vector ###")
    kmeans_pred = run_kmeans(latent, 
                             args.classnum, 
                             random_state=args.seed)

    adata_embedding.obs['label'] = Y
    adata_embedding.obs['label'] = adata_embedding.obs['label'].astype("category")
    # adata_embedding.obs['annotation'] = cell_type
    adata_embedding.obs['annotation'] = np.array(list(map(str, cell_type))) 
    adata_embedding.obs['kmeans'] = kmeans_pred
    adata_embedding.obs['kmeans'] = adata_embedding.obs['kmeans'].astype("category")

    return adata_embedding, Y, leiden_pred, kmeans_pred, val_loader


def validate(args,
             model, 
             x,
             y, 
             device, 
             epoch,
             cell_type,
             seed, 
             dataset_name, 
             n_clusters, 
             resolution=0.6, 
             dir_path_name="validation_pics"):
    print("### Creating features from model ###")
    # Y, latent_vector = inference(data_loader, model, device)
    Y, latent_vector, _ = predict(args, model, x, y, device)
    n_clusters = len(np.unique(Y))
    
    print("### Performming Kmeans clustering method on latent vector ###")
    kmeans_pred = run_kmeans(latent_vector=latent_vector, 
                             n_clusters=n_clusters,
                             random_state=seed)
    
    print("### Performming Leiden clustering method on latent vector ###")
    adata, leiden_pred = run_leiden(latent_vector=latent_vector, 
                                    resolution=resolution)
    adata.obs['cell_type'] = cell_type
    adata.obs['cell_type'] = adata.obs['cell_type'].astype('category')

    # Evaluation 
    k_nmi, k_ari, k_f, k_acc = evaluate_kmeans(Y, kmeans_pred)
    nmi_leiden, ari_leiden, f_leiden = evaluate_leiden(Y, leiden_pred)

    # Visualization
    dir_path_name = f"{dir_path_name}_seed_{seed}"
    embedding_cluster_visualization_sc(adata, 
                                       dataset_name,
                                       Y,
                                       kmeans_pred,
                                       k_nmi,
                                       k_ari,
                                       nmi_leiden,
                                       ari_leiden, 
                                       epoch,
                                       seed,
                                       dir_path_name=dir_path_name)
    # Print out related results
    print(f"### Epoch {epoch} Results: ###")

    tb = pt.PrettyTable()
    tb.field_names = ["Method Name", "NMI", "ARI", "F", "Accuracy"]
    
    tb.add_row(["KMeans", round(k_nmi, 4), round(k_ari, 4), round(k_f, 4), round(k_acc, 4)])
    tb.add_row(["Leiden", round(nmi_leiden, 4), round(ari_leiden, 4), round(f_leiden, 4), 0.0])

    print(tb)
    
    # Write out the evaluation results
    result_dir = f"validation_{seed}"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    file_name = os.path.join(result_dir, f"{dataset_name}.txt")
    f = open(file_name, "a")
    current_time = time.strftime('%Y_%m_%d %H_%M_%S',time.localtime(time.time()))

    f.write(f"Epoch {epoch} -------> {current_time}\n")
    f.write(str(tb) + '\n')
    f.close()


