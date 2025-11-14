import torch.nn as nn
import torch.nn.functional as F
import torch
import resnet
import torchvision.models as torchmodel
from utils import *
#import torch_scatter
import copy

DATASETS_CLASSES = {
               'cifar100':100,
               'cub':200,
               'aircraft':100
               }

class SimpleCNN(nn.Module):
    '''Source: https://github.com/VSainteuf/metric-guided-prototypes-pytorch'''
    def __init__(self, output_dim = 10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1,16,3),nn.BatchNorm2d(16),nn.ReLU(),nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(nn.Conv2d(16,16,3),nn.BatchNorm2d(16),nn.ReLU(),nn.MaxPool2d(2))
        self.conv3 = nn.Sequential(nn.Conv2d(16,32,3),nn.BatchNorm2d(32),nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(32, 64),nn.Linear(64,output_dim))
        
    def forward(self, x):
        out = self.conv3(self.conv2(self.conv1(x)))
        out = out.view(out.shape[0],out.shape[1], -1).max(-1)[0]
        out = self.fc(out)
        return out
   
class metric_model(nn.Module):
    def __init__(
        self,
        model,
        device: str = 'cuda',
        output_dim: int = 8,
        temperature: float = 1.0,
        dataset: str = 'mnist',
        geometry: str = 'euclidean',
        shrink_init: bool = False,
        epsilon: float = 0.1
    ):
        super(metric_model, self).__init__()
        self.model = model
        self.device = device
        self.output_dim = output_dim
        self.temperature = temperature
        self.dataset = dataset
        self.geometry = geometry
        self.shrink_init = shrink_init
        self.epsilon = epsilon

        self.manifold = Manifold(self.geometry)
        self.num_classes = DATASETS_CLASSES[self.dataset]
        if self.shrink_init:
            prototypes = nn.Parameter(
                (2 * torch.rand((DATASETS_CLASSES[self.dataset], self.output_dim), device=self.device) - 1) * self.epsilon, requires_grad=True)
        else:
            prototypes = nn.Parameter(
                torch.rand((DATASETS_CLASSES[self.dataset], self.output_dim), device = self.device), requires_grad = True)
        if self.geometry == 'hyperspherical':
            self.parametric_prototypes = nn.Parameter(F.normalize(prototypes, p=2, dim=1), requires_grad=True)
        else:
            self.parametric_prototypes = geoopt.ManifoldParameter(self.manifold.project(prototypes), manifold=self.manifold.manifold, requires_grad=True)

        self.centroid_prototypes = geoopt.ManifoldParameter(self.manifold.project(prototypes), manifold = self.manifold.manifold, requires_grad=False)
        self.prototypes = self.parametric_prototypes
        self.counter = torch.zeros(DATASETS_CLASSES[self.dataset])

    def forward(self, images):
        embeddings = self.model(images)

        #--Poincaré
        if self.geometry == 'poincare':
            embeddings = clip(embeddings, 1)
            embeddings = self.manifold.project(embeddings)

        #--NormFace
        elif self.geometry == 'hyperspherical':
            # normalize embeddings and prototypes
            embeddings = F.normalize(embeddings, p=2, dim=1)
            self.prototypes.data = F.normalize(self.prototypes.data, p=2, dim=1)
            # cosine similarity
            dists = nn.CosineSimilarity(dim=-1)(embeddings[:, None, :], 
                                                self.prototypes[None, :, :])
            return dists / self.temperature, embeddings

        #--Euclidean/Lorentz
        else:
            embeddings = self.manifold.project(embeddings)
        
        if not self.manifold.manifold._check_point_on_manifold(self.prototypes, atol=1e-3, rtol=1e-3)[0]:
            print(self.manifold.manifold._check_point_on_manifold(self.prototypes))
            print('projecting the prototypes')
            self.prototypes.data = self.manifold.project(self.prototypes.data)
            print(self.manifold.manifold._check_point_on_manifold(self.prototypes))
            print('')

        dists = self.manifold.distance(embeddings, self.prototypes)

        return -dists/self.temperature, embeddings
    
    @torch.no_grad()
    def calculate_centroid_prototypes(self, embeddings: torch.tensor, y:torch.tensor):
        represented_classes = torch.unique(y).detach().cpu().numpy()
        
        # Compute Prototypes
        new_prototypes = torch_scatter.scatter_mean(
            embeddings, y.unsqueeze(1), dim=0, dim_size=self.num_classes
        ).detach()
        
        self.centroid_prototypes[represented_classes, :] = (
            self.counter[represented_classes, None]
            * self.centroid_prototypes[represented_classes, :]
            + new_prototypes[represented_classes, :]
        ) / (self.counter[represented_classes, None] + 1)
        self.counter[represented_classes]
        self.counter[represented_classes] = self.counter[represented_classes] + 1
        
def clip(input_vector, r):
    input_norm = torch.norm(input_vector, dim = -1)
    min_norm = torch.clamp(float(r)/input_norm, max = 1)
    return min_norm[:, None] * input_vector

def load_backbone(config):
    model = config['model']
    modello = model.lower()

    if modello == 'simplecnn':
        out_model = SimpleCNN(output_dim=config['output_dim'])

    elif modello == 'resnet18':
        dataset_name = config['dataset']['name']
        reduced = config['dataset'].get('reduced', False)

        if dataset_name in ['cifar10', 'cifar100']:
            if reduced:
                print('Using pretrained ResNet18 for CIFAR')
                out_model = torchmodel.resnet18(weights='DEFAULT', progress=True)
                # Patch conv1 and maxpool for CIFAR
                out_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                out_model.maxpool = nn.Identity()
            else:
                out_model = resnet.resnet18()  #not pretrained

        else:
            if reduced:
                print('Using pretrained ResNet18')
                out_model = torchmodel.resnet18(weights='DEFAULT', progress=True)
            else:
                out_model = torchmodel.resnet18()

        num_ftrs = out_model.fc.in_features
        out_model.fc = nn.Linear(num_ftrs, config['output_dim'])

    else:
        raise Exception('Selected model is not available.')

    return out_model
