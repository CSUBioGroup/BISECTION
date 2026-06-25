import torch
from torch import nn
import torch.nn.functional as F
 
class Prunable_Linear(nn.Linear):
    def __init__(self, in_features, out_features, device="cpu"):
        super().__init__(in_features, out_features)
        self.in_features = in_features
        self.out_features = out_features
        self.prune_mask = torch.ones_like(self.weight)
        self.flag = False
        self.device = device
        
    def forward(self, x):
        if not self.flag:
            weight = self.weight.to(self.device)
        else:
            self.weight = self.weight.to(self.device)
            self.prune_mask = self.prune_mask.to(self.device)
            weight = self.weight * self.prune_mask
        
        return F.linear(x, weight)
    
    def set_prune_flag(self, flag=False):
        self.flag = flag

class Prunable_Encoder(nn.Module):
    def __init__(self, 
                 in_features, 
                 num_cluster,
                 latent_features=[1024, 512, 128], 
                 device="cpu", 
                 p=0.0):
        super().__init__()
        self.in_features = in_features
        self.latent_features = latent_features
        self.device = device
        
        layers = []
        layers.append(nn.Dropout(p=p))
        for i in range(len(latent_features)):
            if i == 0:
                layers.append(Prunable_Linear(in_features, latent_features[i], device=self.device))
                layers.append(nn.ReLU())
            else:
                layers.append(Prunable_Linear(latent_features[i-1], latent_features[i], device=self.device))
                layers.append(nn.ReLU())

        layers = layers[:-1]
        
        self.prunable_enc = nn.Sequential(*layers)

        # if not mlp fc as a cluster projection if not mlp
        self.fc = nn.Linear(latent_features[-1], num_cluster)
        # self.fc = nn.Sequential(nn.Linear(latent_features[-1], num_cluster),
        #                         nn.Softmax(dim=1))
        
    def forward(self, x):
        h = self.prunable_enc(x)
        out = self.fc(h)
        
        return out
    
    def get_embedding(self, x):
        self.set_prune_flag(flag=False)
                
        latent = self.prunable_enc(x)
        
        return latent
    
    def set_prune_flag(self, flag=False):
        for name, module in self.named_modules():
            if isinstance(module, Prunable_Linear):
                module.set_prune_flag(flag)

class ProjectionHeader(nn.Module):
    def __init__(self, rep_dim, device="cpu"):
        super().__init__()
        self.proj_header = nn.Sequential(nn.Linear(in_features=rep_dim, out_features=rep_dim, device=device),
                                         nn.ReLU(),
                                         nn.Linear(in_features=rep_dim, out_features=rep_dim, device=device))

    def forward(self, x):
        out = self.proj_header(x)
        
        return out
