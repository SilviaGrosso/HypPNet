
import torch
import pickle as pkl
import torch.optim as optim
import geoopt
import torch.nn as nn


DATASETS_CLASSES = {
               'cifar100':100,
               'cub':200,
               'aircraft':100
               }

class SeparationLoss(nn.Module):
    '''
    Source: https://github.com/VSainteuf/metric-guided-prototypes-pytorch
    Large margin separation between hyperspherical protoypes
    '''

    def __init__(self):
        super(SeparationLoss, self).__init__()

    def forward(self, protos):
        '''
        Args:
            protos (tensor): (N_prototypes x Embedding_dimension)
        '''
        M = torch.matmul(protos, protos.transpose(0, 1)) - 2 * torch.eye(
            protos.shape[0]
        ).to(protos.device)
        return M.max(dim=1)[0].mean()


def hyperspherical_embedding(dataset, device, embedding_dimension, seed):
    '''
    Source: https://github.com/VSainteuf/metric-guided-prototypes-pytorch.
    Function to learn the prototypes according to the separationLoss Minimization
    embedding_dimension : 
    We use SGD as optimizer
    lr : learning rate
    momentum : momentum
    n_steps : number of steps for learning the prototypes
    wd : weight decay
    '''
    lr=0.1
    momentum=0.9
    n_steps=1000
    wd=1e-4

    torch.manual_seed(seed) 
    mapping = torch.rand((DATASETS_CLASSES[dataset], embedding_dimension), device=device, requires_grad = True)
    
    optimizer = torch.optim.SGD([mapping], lr=lr, momentum=momentum, weight_decay=wd)
    L_hp = SeparationLoss()
    
    for i in range(n_steps):
        with torch.no_grad():
            mapping.div_(torch.norm(mapping, dim=1, keepdim=True))
        optimizer.zero_grad()
        loss = L_hp(mapping)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        mapping.div_(torch.norm(mapping, dim=1, keepdim=True))
    return mapping.detach()

    

def load_cost_matrix(dataset_name):
    basedir = f"./Datasets_features/{dataset_name}"
    dataset_name = dataset_name.lower()
    if dataset_name == 'mnist':
        return torch.Tensor(pkl.load(open(basedir + f'/{dataset_name}.pkl','rb')))[:10,:][:,:10]    
    return torch.Tensor(pkl.load(open(basedir + f'/{dataset_name}.pkl','rb')))
    

def load_optimizer(params, *args):
    optimname = args[0].lower()
    learning_rate = args[1]
    decay = args[2]
    momentum = args[3]
    
    if optimname == "sgd":
        optimizer = optim.SGD(params, lr=learning_rate, momentum=momentum, weight_decay=decay)
    elif optimname == "adam":
        optimizer = optim.AdamW(params, lr=learning_rate, weight_decay=decay)
    elif optimname == "rsgd":
        optimizer = geoopt.optim.RiemannianSGD(params, lr=learning_rate, weight_decay=decay, stabilize=1)
    elif optimname == "radam":
        optimizer = geoopt.optim.RiemannianAdam(params, lr=learning_rate, weight_decay=decay, stabilize=1)
    return optimizer


class Manifold(nn.Module):
    def __init__(self, geometry):
        super(Manifold, self).__init__()
        self.geometry = geometry
        if geometry == 'poincare':
            self.manifold = geoopt.PoincareBallExact(c=1, learnable = False)
        if geometry == 'lorentz':
            self.manifold = geoopt.Lorentz()
        elif geometry == 'hyperspherical':
            self.manifold = geoopt.SphereExact()
        elif geometry == 'euclidean':
            self.manifold = geoopt.Euclidean()

    def project(self, x):
        if self.geometry == 'euclidean':
            return x
        elif self.geometry == 'poincare':
            return self.manifold.expmap0(x)
        elif self.geometry in ['hyperspherical', 'lorentz']:
            return self.manifold.projx(x)
        
        
    def distance(self, x, p):
        """
        This function returns the distance between each x and each p.
        """
        if self.geometry == 'euclidean':
            return torch.norm(x[:, None, :] - p[None, :, :], dim=-1)
        elif self.geometry in ['poincare', 'lorentz']:
            return self.manifold.dist(x[:, None, :], p[None, :, :])
        elif self.geometry == 'hyperspherical':
            return (1 - nn.CosineSimilarity(dim=-1)(x[:, None, :], p[None, :, :]))     
     
