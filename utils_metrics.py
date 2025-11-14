from typing import Tuple
import torch
import torch
import torch.nn.functional as F
from torchvision import transforms
import torchattacks
import logging
from data import load_dataset
import torch.nn as nn
from torchattacks.attack import Attack
from sklearn.metrics import roc_auc_score

mean_dict = {
    'cifar10': [0.4914, 0.4822, 0.4465],
    'cifar100': [0.507, 0.487, 0.441],
    'cub' : [0.485, 0.456, 0.406],
    'aircraft' : [0.485, 0.456, 0.406],
    'cars' : [0.485, 0.456, 0.406],
    'mnist' : [0.13066062]
}

std_dict = {
    'cifar10' : [0.2023, 0.1994, 0.2010],
    'cifar100': [0.267, 0.256, 0.276],
    'cub': [0.229, 0.224, 0.225],
    'aircraft': [0.229, 0.224, 0.225],
    'cars': [0.229, 0.224, 0.225],
    'mnist' : [0.30810776]
}

DIMENSION = {
    'cifar100':64,
    'cifar10':8,
    'cub':128,
    'aircraft':64,
    'cars':128
}


class custom_PGD(Attack):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of iterations. (Default: 40)

    Shape:
        - images: (N, C, H, W) with values in [0, 1]
        - labels: (N)

    Example::
        >>> attack = custom_PGD(model, eps=8/255, alpha=2/255, steps=40)
        >>> adv_images = attack(images, labels)
    """

    def __init__(self, model, eps=8/255, alpha=2/255, steps=40):
        super().__init__("PGD", model)
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.supported_mode = ["default", "targeted"]

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        adv_images = images + torch.empty_like(images).uniform_(-self.eps, self.eps)
        adv_images = torch.clamp(adv_images, 0, 1).detach()

        loss = nn.CrossEntropyLoss()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs, _ = self.model(adv_images)
            # outputs = F.softmax(outputs, dim=1)

            if self.targeted:
                target_labels = self.get_target_label(images, labels)
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            grad = torch.autograd.grad(
                cost, adv_images, retain_graph=False, create_graph=False
            )[0]

            adv_images = adv_images + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images


def denorm(batch, mean=[0.1307], std=[0.3081], device = 'cpu'):
    """
    Convert a batch of tensors to their original scale.

    Args:
        batch (torch.Tensor): Batch of normalized tensors.
        mean (torch.Tensor or list): Mean used for normalization.
        std (torch.Tensor or list): Standard deviation used for normalization.

    Returns:
        torch.Tensor: batch of tensors without normalization applied to them.
    """
    if isinstance(mean, list):
        mean = torch.tensor(mean).to(device)
    if isinstance(std, list):
        std = torch.tensor(std).to(device)

    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)

def robustness(model, device, test_loader, epsilon, dataset):
    correct = 0
    den = 0
    mean = mean_dict[dataset]
    std = std_dict[dataset]
    atk = custom_PGD(model, eps=epsilon, alpha=epsilon/2, steps=40)


    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data_denorm = denorm(data, mean=mean, std=std, device=device)
        adv_images = atk(data_denorm, target)
        adv_images = transforms.Normalize(mean, std)(adv_images)
        dists, _ = model(adv_images)
        final_pred = dists.max(-1, keepdim=True)[1].squeeze()
        correct += (final_pred == target).sum().item()
        den += len(target)


    final_acc = correct / den
    print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {den} = {final_acc}")
    return final_acc, None



def get_robustness(test_loader, model, config):
    epsilons = [0, 0.8/255, 1.6/255, 3.2/255]

    PGD_accuracies = []

    for eps in epsilons:     
        acc, _ = robustness(model, config['device'], test_loader, eps, config['dataset']['name'])
        PGD_accuracies.append(acc)

    return PGD_accuracies

    
@torch.inference_mode()
def compute_confidences(model, device, dataloader):
    model.eval()
    confidences = []
    for data, _ in dataloader:
        data = data.to(device)
        outputs = model(data)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        probs = F.softmax(outputs, dim=1)
        scores = probs.max(dim=1)[0]
        confidences.append(scores.cpu())
    return torch.cat(confidences).numpy()


def get_OOD_and_AUROC(model, config, return_distributions=False):
    device = config['device']
    dataset_name = config['dataset']['name'].lower()

    ood_mapping = {
        'cifar10': 'cifar100',
        'cifar100': 'cifar10',
        'aircraft': 'cub',
        'cub': 'aircraft',
        'cars': 'aircraft'
    }

    if dataset_name not in ood_mapping:
        raise ValueError(f"No OOD dataset mapping defined for {dataset_name}")

    ood_dataset = ood_mapping[dataset_name]
    print(f"Evaluating OOD + AUROC for {dataset_name} vs {ood_dataset}...")

    _, id_loader, _ = load_dataset(dataset_name, batch_size=128, num_workers=4)
    _, ood_loader, _ = load_dataset(ood_dataset, batch_size=128, num_workers=4)

    id_conf = compute_confidences(model, device, id_loader)
    ood_conf = compute_confidences(model, device, ood_loader)

    labels = [1] * len(id_conf) + [0] * len(ood_conf)
    scores = list(id_conf) + list(ood_conf)
    auroc = roc_auc_score(labels, scores)

    id_mean = float(torch.tensor(id_conf).mean())
    ood_mean = float(torch.tensor(ood_conf).mean())

    print(f"AUROC: {auroc:.5f} | ID_mean: {id_mean:.5f} | OOD_mean: {ood_mean:.5f}")

    results = {
        'ID_dataset': dataset_name,
        'OOD_dataset': ood_dataset,
        'AUROC': auroc,
        'ID_conf_mean': id_mean,
        'OOD_conf_mean': ood_mean
    }

    if return_distributions:
        results['ID_confidences'] = id_conf.tolist()
        results['OOD_confidences'] = ood_conf.tolist()


    return results
