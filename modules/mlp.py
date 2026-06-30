import torch
from torch import nn
import torch.nn.functional as F


class KnowledgeLinear(nn.Linear):
    def __init__(self, in_features, out_features, pathway, device='cpu'):
        super().__init__(in_features, out_features)
        self.in_features = in_features
        self.out_features = out_features
        self.pathway = pathway
        self.device = device
    
    def forward(self, x):
        if self.pathway is not None:
            self.weight = self.weight.to(self.device)
            self.pathway = self.pathway.to(self.device)
            weight = self.weight * self.pathway
        else:
            weight = self.weight.to(self.device)
        
        return F.linear(x, weight=weight)    
        

class Encoder(nn.Module):
    def __init__(self,
                 in_features,
                 num_cluster,
                 latent_features = [1024, 512, 128],
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
                layers.append(nn.Linear(in_features, latent_features[i]))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Linear(latent_features[i-1], latent_features[i]))
                layers.append(nn.ReLU())
        
        layers = layers[:-1]
        self.encoder = nn.Sequential(*layers)

        self.fc = nn.Linear(latent_features[-1], num_cluster)
        
    def forward(self, x):
        h = self.encoder(x)
        out = self.fc(h)

        return out
    
    def get_embedding(self, x):
        latent = self.encoder(x)

        return latent    

class KnowledgeEncoder(nn.Module):
    def __init__(self, 
                 in_features, 
                 num_cluster, 
                 latent_features, 
                 device="cpu", 
                 p=0.0,
                 pathway=None): 
        super().__init__()
        self.in_features = in_features
        self.latent_features = latent_features
        self.pathway = pathway
        if self.pathway is not None:
            self.latent_features[0] = pathway.shape[0]
            
        self.device = device
        
        layers = []
        layers.append(nn.Dropout(p=p))
        for i in range(len(self.latent_features)):
            if i == 0:
                layers.append(KnowledgeLinear(in_features=self.in_features,
                                              out_features=self.latent_features[0],
                                              pathway=self.pathway,
                                              device=self.device))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Linear(in_features=self.latent_features[i-1],
                                        out_features=self.latent_features[i]))
                layers.append(nn.ReLU())
                
        layers = layers[:-1]
        self.encoder = nn.Sequential(*layers)
        self.fc = nn.Linear(self.latent_features[-1], num_cluster)
        
    def forward(self, x):
        h = self.encoder(x)
        out = self.fc(h)
        
        return out
    
class KnowledgeStudent(nn.Module):
    def __init__(self,
                 in_features,
                 num_cluster,
                 pathway=None,
                 latent_features = [1024, 512, 128],
                 device="cpu",
                 p=0.0):
        super().__init__()
        self.in_features = in_features
        self.latent_features = latent_features
        self.device = device
        self.pathway = pathway

        self.attention = torch.nn.Linear(self.in_features, self.in_features)
        self.encoder = KnowledgeEncoder(in_features=self.in_features,
                                        num_cluster=num_cluster,
                                        latent_features=self.latent_features,
                                        device=self.device,
                                        p=p,
                                        pathway=self.pathway)
    
    def forward(self, x):
        outputs = self.encoder(x)

        return outputs