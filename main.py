import argparse
import itertools
import os
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

import numpy as np
import prettytable as pt
import torch
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from torch import nn
from torch.utils.data import DataLoader

from evaluation import plot
from loaders import CellDataset, CellDatasetPseudoLabel, get_anchor, prepareForKD
from modules import DistillerLoss, Encoder, KnowledgeStudent
from train import get_pseudo_labels, pretrain
from utils import (
    getGeneSetMatrix,
    get_prediction,
    get_prediction_with_gene_set,
    set_seed,
    train_distiller_with_gene_set,
    train_teacher,
    yaml_config_hook,
)

PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_project_path(path_value):
    if path_value is None:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str((PROJECT_ROOT / path).resolve())


def select_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def main(dname):
    parser = argparse.ArgumentParser()
    config_path = PROJECT_ROOT / "config_test_att" / f"config_{dname}.yaml"
    config = yaml_config_hook(str(config_path))

    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, cuda:0, etc.")
    parser.add_argument(
        "--embedding_dir",
        default="./outputs/embeddings",
        help="Directory used to save final h5ad embeddings.",
    )

    args = parser.parse_args()
    args.root_dir = resolve_project_path(args.root_dir)
    args.gene_sets_path = resolve_project_path(args.gene_sets_path)
    args.model_path = resolve_project_path(args.model_path)
    args.embedding_dir = resolve_project_path(args.embedding_dir)

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    print(f"Current Random seed {args.seed}")
    print(f"Current Flag: {args.flag}")

    set_seed(args.seed)

    device = select_device(args.device)
    print(device)
    # start_time1 = time.time()
    # 1. pretrain 
    print("---------- Step 1: Pretrain Model ----------")
    data, label, cell_type, model, start_time1 = pretrain(args, device=device)

    # 2. get pseudo label 
    print('---------- Step 2: Get Pseudo Labels ----------')
    adata_embedding, Y, leiden_pred, _, _ = get_pseudo_labels(args, 
                                                              model, 
                                                              data, 
                                                              label, 
                                                              cell_type, 
                                                              device)
    
    end_time1 = time.time()
    plot(adata_embedding, 
         Y, 
         args.dataset_name, 
         epoch=args.epochs, 
         seed=args.seed,
         dir_path_name="pictures")
    
    # 3. train teacher model
    # 3.1 data prepare
    num_genes = 5000
    x, y, adata_new, hvg_new = prepareForKD(root_dir=args.root_dir,
                                            data_type=args.data_type,
                                            dataset_name=args.dataset_name,
                                            num_genes=num_genes,
                                            scale=False)

    adata_new.obs.loc[:, 'leiden'] = adata_embedding.obs.loc[:, 'leiden'].values
    adata_new.obs.loc[:, 'kmeans'] = adata_embedding.obs.loc[:, 'kmeans'].values
    adata_new.obs.loc[:, 'label'] = adata_embedding.obs.loc[:, 'label'].values

    print('-' * 40)
    print(adata_new)
    
    print('---------- Step 3: Train & Evaluate Teacher Model ----------')
    adata_new, adata_embedding = get_anchor(adata_new, 
                                            adata_embedding, 
                                            pseudo_label='leiden',
                                            k=30, 
                                            percent=0.5)

    new_val_datasets = CellDataset(x, y)
    in_features = new_val_datasets.data.size(0)
    new_val_loader = DataLoader(new_val_datasets,
                                batch_size=256,
                                shuffle=False,
                                num_workers=args.workers,
                                drop_last=False)
    
    train_adata = adata_new[adata_new.obs.leiden_density_status == 'low', :].copy()
    test_adata = adata_new[adata_new.obs.leiden_density_status == 'high', :].copy()

    pseudo_labels = np.array(list(map(int, train_adata.obs['leiden'].values)))
    print(f"extracted_nmi: {normalized_mutual_info_score(train_adata.obs['Group'].values, pseudo_labels):.4f}")
    print(f"extracted_ari : {adjusted_rand_score(train_adata.obs['Group'].values, pseudo_labels):.4f}")

    train_dataset = CellDatasetPseudoLabel(train_adata, 
                                           pseudo_label='leiden', 
                                           oversample_flag=True)
    test_dataset = CellDatasetPseudoLabel(test_adata,
                                          pseudo_label='leiden', 
                                          oversample_flag=False)

    print(f"teacher train dataset: {len(train_dataset)}")
    print(f"teacher test dataset: {len(test_dataset)}")
    
    if hvg_new is None:
        gene_set_matrix = None
    else:
        gene_sets_path = args.gene_sets_path
        gene_sets_name = sorted(os.listdir(gene_sets_path))
        print(gene_sets_name)

        _matrix_list = []
        _keys_list = []
        _df_list = {}
        for name in gene_sets_name:
            if name.endswith(".gmt"):
                _matrix, _keys, _df = getGeneSetMatrix(name, hvg_new, gene_sets_path)
                _matrix_list.append(_matrix)
                _keys_list.append(_keys)
                _df_list.update(_df)
            # if name.endswith(".txt"):
            #     _matrix, _keys = getGeneSetMatrix(name, hvg, gene_sets_path)
            #     _matrix_list.append(_matrix)
            #     _keys_list.append(_keys)

        gene_set_matrix = np.concatenate(_matrix_list, axis=0)
        keys_all = list(itertools.chain(*_keys_list))

        gene_set_matrix = torch.from_numpy(gene_set_matrix).to(device).float()
        print(f"pathway shape: {gene_set_matrix.size()}")
        print('-' * 40)
        print(f"sum = {gene_set_matrix.sum()}")

    # 3.2 build KD model
    in_features = adata_new.shape[1]
    teacher = Encoder(in_features=in_features,
                      num_cluster=len(np.unique(leiden_pred)),
                      latent_features=args.latent_feature,
                      device=device,
                      p=args.p)

    student = KnowledgeStudent(in_features=in_features,
                               num_cluster=len(np.unique(leiden_pred)),
                               pathway=gene_set_matrix,
                               latent_features=args.latent_feature,
                               device=device,
                               p=args.p)
    
    # 3.3 loader pretrained weight for teacher model
    train_loader = DataLoader(train_dataset, 
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.workers)
    
    teacher_criterion = nn.CrossEntropyLoss()   
    teacher_optimizer = torch.optim.Adam(teacher.parameters(), 
                                     lr=args.learning_rate, 
                                     weight_decay=0.0)
    
    # 4. train teacher model
    start_time2 = time.time()
    teacher_epochs = args.teacher_epochs
    for epoch in range(args.start_epoch, teacher_epochs+1):
        loss_epoch = train_teacher(train_loader, 
                                   teacher, 
                                   teacher_criterion, 
                                   teacher_optimizer, 
                                   device, 
                                   teacher_epochs)

        print(f"Epoch [{epoch}/{teacher_epochs}]\t Loss: {loss_epoch}")
        print('-' * 60)

    # 4.2 evaluation tearcher model performance
    teacher_pred = get_prediction(teacher, device, new_val_loader, name="teacher")
    teacher_pred = np.array(teacher_pred, dtype=np.int32)
    adata_embedding.obs['teacher_prediction'] = teacher_pred
    adata_embedding.obs['teacher_prediction'] = adata_embedding.obs['teacher_prediction'].astype('category')

    teacher_ari = adjusted_rand_score(Y, teacher_pred)
    teacher_nmi = normalized_mutual_info_score(Y, teacher_pred)

    leiden_ari = adjusted_rand_score(Y, leiden_pred)
    leiden_nmi = normalized_mutual_info_score(Y, leiden_pred)

    print(f"---- teacher ari: {teacher_ari} ----")
    print(f"---- teacher nmi: {teacher_nmi} ----")
   
    # 5. train distiller
    print('---------- Step 4: Train & Evaluate Distiller ----------')
    distiller_loss = DistillerLoss(alpha=args.kd_alpha, 
                                   temperature=args.kd_temperature)
    distiller_optimizer = torch.optim.Adam(student.parameters(), 
                                           lr=args.learning_rate, 
                                           weight_decay=0.0)

    # freeze parameters in teacher model
    for name, param in teacher.named_parameters():
        param.requires_grad = False
            
    distiller_epochs = args.distiller_epochs
    for epoch in range(args.start_epoch, distiller_epochs+1):
        loss_epoch = train_distiller_with_gene_set(train_loader, 
                                                   student, 
                                                   teacher, 
                                                   distiller_loss, 
                                                   distiller_optimizer, 
                                                   device=device)

        print(f"Epoch [{epoch}/{distiller_epochs}]\t Loss: {loss_epoch}")
        print('-' * 60)
        
    end_time2 = time.time()
    
    # 5.2 evalation student model
    student_pred = get_prediction_with_gene_set(student, 
                                                device, 
                                                new_val_loader, 
                                                name="student")
    student_pred = np.array(student_pred, dtype=np.int32)
    adata_embedding.obs['student_prediction'] = student_pred
    adata_embedding.obs['student_prediction'] = adata_embedding.obs['student_prediction'].astype("category")

    student_ari = adjusted_rand_score(Y, student_pred)
    student_nmi = normalized_mutual_info_score(Y, student_pred)

    # save result 
    tb = pt.PrettyTable()
    tb.field_names = ['Method Name', 'ARI', 'NMI', 'time']
    time1 = end_time1 - start_time1
    time2 = end_time2 - start_time2
    tb.add_row(['Leiden', round(leiden_ari, 4), round(leiden_nmi, 4), time1])
    tb.add_row(['Teacher', round(teacher_ari, 4), round(teacher_nmi, 4), time2])
    tb.add_row(['Student', round(student_ari, 4), round(student_nmi, 4), time1 + time2])

    print(tb)

    result_dir = f"validation_{args.seed}"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    file_name = os.path.join(result_dir, f"{args.dataset_name}.txt")
    f = open(file_name, "a")
    current_time = time.strftime("%Y_%m_%d %H_%M_%S", time.localtime(time.time()))

    f.write(f"Epoch {args.epochs} -----------> {current_time}\n")
    f.write(str(tb) + '\n')
    f.close()

    # 6. visualize result
    plot(adata_embedding, 
         Y, 
         args.dataset_name, 
         args.epochs, 
         seed=args.seed, 
         colors=['student_prediction', 'annotation'],
         titles=['Student Prediction', 'Cell Type'],
         dir_path_name="pictures")

    plot(adata_embedding, 
         Y, 
         args.dataset_name, 
         args.epochs, 
         seed=args.seed, 
         colors=['teacher_prediction', 'student_prediction'],
         titles=['Teacher Prediction', 'Student Prediction'],
         dir_path_name="pictures")

    # 7. save embeddings
    if not os.path.exists(args.embedding_dir):
        os.makedirs(args.embedding_dir)
    write_path = os.path.join(args.embedding_dir, f"{dname}_{args.seed}.h5ad")
    adata_embedding.write_h5ad(write_path)


if __name__ == "__main__":

    dname = "Pancreas"

    main(dname)






