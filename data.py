from cub2011 import Cub2011
from aircraft import Aircraft
from collections import Counter
import torch.utils.data as data
from torchvision import datasets, transforms
import numpy as np
import torch

DATASET_N_CLASSES = {
    'mnist': 10,
    'cifar10': 10,
    'cifar100': 100,
    'cub': 200,
    'aircraft': 100,
    'cars': 196
}


def load_dataset(dataset_name:str, 
                 batch_size:int, 
                 num_workers:int = 0, 
                 val:bool = False, 
                 reduced:bool = False, 
                 ex_4_class:int = 0):
    dataset_name = dataset_name.lower()
    print(f'loading the {dataset_name} dataset')
    if dataset_name == 'cifar100':
        return load_cifar100(batch_size, num_workers, reduced, ex_4_class)
    elif dataset_name == 'cifar10':
        return load_cifar10(batch_size, num_workers, reduced, ex_4_class)
    elif dataset_name == "mnist":
        return load_MNIST(batch_size, num_workers, reduced, ex_4_class)
    elif dataset_name == "cub":
        return load_cub(batch_size, num_workers, reduced, ex_4_class)
    elif dataset_name == "cars":
        return load_cars(batch_size, num_workers, reduced, ex_4_class)
    elif dataset_name == "aircraft":
        return load_aircraft(batch_size, num_workers, reduced, ex_4_class)
    else:
        raise Exception('Selected dataset is not available.')

def load_cifar100(batch_size, num_workers, reduced=False, ex_4_class=0):

    mrgb = [0.507, 0.487, 0.441]
    srgb = [0.267, 0.256, 0.276]
    size = transforms.Normalize(mean=mrgb, std=srgb)

    transformations=transforms.Compose([
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=mrgb, std=srgb)])
        
    test_transformations = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(mean=mrgb, std=srgb)])
    
    train = datasets.CIFAR100('data/cifar100/', train = True, transform = transformations, download = True)
    test = datasets.CIFAR100('data/cifar100/', train = False, transform = test_transformations, download = True)
    
    num_train = len(train)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    
    valid_size = 0.1
    split = int(np.floor(valid_size * num_train))
    train_indices, valid_indices = indices[split:], indices[:split]
    validation = torch.utils.data.Subset(train, valid_indices)
    train = torch.utils.data.Subset(train, train_indices)
    # test = torch.utils.data.Subset(test, np.arange(1000)[:100])
    validloader = data.DataLoader(validation, batch_size=256, num_workers=num_workers, shuffle = True, drop_last=False)
    
    if reduced:
        class_counts = {i: 0 for i in range(DATASET_N_CLASSES['cifar100'])} 
        reduced_indices = []
        for idx in np.arange(len(train_indices)):
            label = train.dataset[idx][1]  
            if class_counts[label] < ex_4_class:
                reduced_indices.append(idx)
                class_counts[label] += 1
            if all(count >= ex_4_class for count in class_counts.values()):
                break
        train = torch.utils.data.Subset(train, reduced_indices)
    print('check train size:', len(train))
    trainloader = data.DataLoader(train, batch_size=batch_size, num_workers=num_workers, shuffle = True, drop_last=False)
    testloader = data.DataLoader(test, batch_size=256, num_workers=num_workers)
    
    
    return trainloader, testloader, validloader

def load_cub(batch_size, num_workers, reduced=False, ex_4_class=0):

    transform_train = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomCrop(224, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    transform_test = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    train = Cub2011('data/cub2011', train=True, transform=transform_train, download=True)
    test = Cub2011('data/cub2011', train=False, transform = transform_test, download=True)

    num_train = len(train)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    
    valid_size = 0.1
    split = int(np.floor(valid_size * num_train))
    train_indices, valid_indices = indices[split:], indices[:split]
    validation = torch.utils.data.Subset(train, valid_indices)
    train = torch.utils.data.Subset(train, train_indices) #25-29 examples per class
    validloader = data.DataLoader(validation, batch_size=256, num_workers=num_workers, shuffle = True, drop_last=False)
    if reduced:
        class_counts = {i: 0 for i in range(DATASET_N_CLASSES['cub'])} 
        reduced_indices = []
        for idx in train.indices:
            label = train.dataset[idx][1]  
            if class_counts[label] < ex_4_class:
                reduced_indices.append(idx)
                class_counts[label] += 1
            if all(count >= ex_4_class for count in class_counts.values()):
                break
        train = torch.utils.data.Subset(train, reduced_indices)
    print('check train size:', len(train))
    trainloader = data.DataLoader(train, batch_size=batch_size, num_workers=num_workers, shuffle = True, drop_last=False)
    testloader = data.DataLoader(test, batch_size=256, num_workers=num_workers)
    
    return trainloader, testloader, validloader
    

def load_aircraft(batch_size, num_workers, reduced=False, ex_4_class=0):

    transform_train = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomCrop(224, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    transform_test = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])


    train = Aircraft(root='data/aircraft', train = True, transform=transform_train)
    test    = Aircraft(root='data/aircraft', train = False, transform=transform_test)

    num_train = len(train)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    
    valid_size = 0.1
    split = int(np.floor(valid_size * num_train))
    train_indices, valid_indices = indices[split:], indices[:split]
    validation = torch.utils.data.Subset(train, valid_indices)
    train = torch.utils.data.Subset(train, train_indices)
    validloader = data.DataLoader(validation, batch_size=256, num_workers=num_workers, shuffle = True, drop_last=False)
    
    if reduced:
        class_counts = {i: 0 for i in range(DATASET_N_CLASSES['aircraft'])} 
        reduced_indices = []
        for idx in np.arange(len(train_indices)):
            label = train.dataset[idx][1]  
            if class_counts[label] < ex_4_class:
                reduced_indices.append(idx)
                class_counts[label] += 1
            if all(count >= ex_4_class for count in class_counts.values()):
                break
        train = torch.utils.data.Subset(train, reduced_indices)
    print('check train size:', len(train))
    trainloader = data.DataLoader(train, batch_size=batch_size, num_workers=num_workers, shuffle = True, drop_last=False)
    testloader = data.DataLoader(test, batch_size=256, num_workers=num_workers)
    
    return trainloader, testloader, validloader
    
 
    
def load_MNIST(batch_size, num_workers=0, reduced=False, ex_4_class=0):

    transform_train = transforms.Compose([
        transforms.RandomAffine(degrees=40, scale=(1.3,1.3)), 
        transforms.ToTensor(), 
        transforms.Normalize([0.13066062],[0.30810776])
        ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize([0.13066062],[0.30810776])
        ])
    train = datasets.MNIST('data/MNIST', train=True, transform = transform_train, download=True)
    test = datasets.MNIST('data/MNIST', train=False, transform = transform_test, download=True)

    num_train = len(train)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    
    valid_size = 0.1
    split = int(num_train - np.floor(valid_size * num_train))
    train_indices, valid_indices = indices[:split], indices[split:]
    validation = torch.utils.data.Subset(train, valid_indices)
    train = torch.utils.data.Subset(train, train_indices)
    validloader = data.DataLoader(validation, batch_size=256, num_workers=num_workers, shuffle = True, drop_last=False)
    
    if reduced:
        class_counts = {i: 0 for i in range(DATASET_N_CLASSES['mnist'])} 
        reduced_indices = []
        for idx in np.arange(len(train_indices)):
            label = train.dataset[idx][1]  
            if class_counts[label] < ex_4_class:
                reduced_indices.append(idx)
                class_counts[label] += 1
            if all(count >= ex_4_class for count in class_counts.values()):
                break
        train = torch.utils.data.Subset(train, reduced_indices)
    trainloader = data.DataLoader(train, batch_size=batch_size, num_workers=num_workers, shuffle = True, drop_last=False)
    testloader = data.DataLoader(test, batch_size=256, num_workers=num_workers)
    
    return trainloader, testloader, validloader
    

def load_cifar10(batch_size, num_workers=0, reduced=False, ex_4_class=0):

    transform_train = transforms.Compose([
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
                                
    transform_test = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train = datasets.CIFAR10('data/cifar10', train=True, transform = transform_train, download=True)
    test = datasets.CIFAR10('data/cifar10', train=False, transform = transform_test, download=True)
    num_train = len(train)
    indices = list(range(num_train))
    np.random.shuffle(indices)

    valid_size = 0.1
    split = int(np.floor(valid_size * num_train))
    train_indices, valid_indices = indices[split:], indices[:split]
    validation = torch.utils.data.Subset(train, valid_indices)
    train = torch.utils.data.Subset(train, train_indices)
    validloader = data.DataLoader(validation, batch_size=256, num_workers=num_workers, shuffle = True, drop_last=False)

    if reduced:
        class_counts = {i: 0 for i in range(DATASET_N_CLASSES['cifar10'])} 
        reduced_indices = []
        for idx in np.arange(len(train_indices)):
            label = train.dataset[idx][1]  
            if class_counts[label] < ex_4_class:
                reduced_indices.append(idx)
                class_counts[label] += 1
            if all(count >= ex_4_class for count in class_counts.values()):
                break
        train = torch.utils.data.Subset(train, reduced_indices)
    print('check train size:', len(train))
    trainloader = data.DataLoader(train, batch_size=batch_size, num_workers=num_workers, shuffle = True, drop_last=False)
    testloader = data.DataLoader(test, batch_size=256, num_workers=num_workers)

    return trainloader, testloader, validloader