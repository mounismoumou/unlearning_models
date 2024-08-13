import os
import random

import numpy as np
import torch
from torch.utils import data
import torchvision.transforms as T
import torchvision.datasets as datasets


class CIFAR10Dataset(data.Dataset):
    """The CIFAR-10 dataset."""

    def __init__(self, root, split='train', transform=None):
        super().__init__()
        self.split = split
        self.transform = transform

        if split == 'train':
            self.dataset = datasets.CIFAR10(root=root, train=True, download=True, transform=self.transform)
        else:
            self.dataset = datasets.CIFAR10(root=root, train=False, download=True, transform=self.transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return {'image': image, 'label': label}


def get_dataset(batch_size=64, dataset_path=''):
    """Get the CIFAR-10 dataset."""
    
    transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_ds = CIFAR10Dataset(root=dataset_path, split='train', transform=transform)
    train_loader = data.DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)

    transform_val_test = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    held_out = CIFAR10Dataset(root=dataset_path, split='test', transform=transform)
    test_set, val_set = torch.utils.data.random_split(held_out, [0.5, 0.5], generator=torch.Generator().manual_seed(42))
    test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    val_loader = data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)

    # Download the forget and retain index split
    local_path = "forget_idx.npy"
    if not os.path.exists(local_path):
        response = requests.get(
            "https://storage.googleapis.com/unlearning-challenge/" + local_path
        )
        open(local_path, "wb").write(response.content)
    forget_idx = np.load(local_path)

    # Construct indices of retain from those of the forget set
    forget_mask = np.zeros(len(train_ds.dataset.targets), dtype=bool)
    forget_mask[forget_idx] = True
    retain_idx = np.arange(forget_mask.size)[~forget_mask]

    # Split train set into a forget and a retain set
    forget_set = torch.utils.data.Subset(train_ds, forget_idx)
    retain_set = torch.utils.data.Subset(train_ds, retain_idx)

    forget_loader = torch.utils.data.DataLoader(
        forget_set, batch_size=batch_size, shuffle=True, num_workers=2
    )
    forget_loader_no_shuffle = data.DataLoader(forget_set, batch_size=batch_size, shuffle=False, num_workers=2)
    retain_loader = torch.utils.data.DataLoader(
        retain_set, batch_size=batch_size, shuffle=True, num_workers=2, generator=torch.Generator().manual_seed(42)
    )

    # Compute class weights
    class_counts = np.bincount(train_ds.dataset.targets)
    class_weights = [1.0 / count if count > 0 else 1.0 for count in class_counts]
    class_weights_tensor = torch.FloatTensor(class_weights)

    return (
        train_loader,
        val_loader,
        test_loader,
        retain_loader,
        forget_loader,
        forget_loader_no_shuffle,
        class_weights_tensor,
    )


def compute_accuracy_cifar10(
        data_names_list,
        data_loader_list,
        net,
        model_name,
        print_per_class_=True,
        print_=True,
):
    """Compute the accuracy."""
    net.eval()
    accs = {}
    pc_accs = {}
    list_of_classes = list(range(10))
    device = 'cuda' if torch.cuda.is_available() else 'mps'

    with torch.no_grad():
        for name, loader in zip(data_names_list, data_loader_list):
            correct = 0
            total = 0
            correct_pc = [0 for _ in list_of_classes]
            total_pc = [0 for _ in list_of_classes]
            for sample in loader:
                inputs = sample['image'].to(device)
                targets = sample['label'].to(device)

                outputs = net(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                for c in list_of_classes:
                    num_class_c = (targets == c).sum().item()
                    correct_class_c = (
                            ((predicted == targets) * (targets == c)).float().sum().item()
                    )
                    total_pc[c] += num_class_c
                    correct_pc[c] += correct_class_c

            accs[name] = 100.0 * correct / total
            pc_accs[name] = [
                100.0 * c / t if t > 0 else -1.0 for c, t in zip(correct_pc, total_pc)
            ]

    if print_:
        logging.info(f'{model_name} accuracy: ' + ', '.join([f'{name}: {acc:.2f}%' for name, acc in accs.items()]))
    if print_per_class_:
        for name in data_names_list:
            logging.info(f'{name} accuracy per class: ' + ', '.join([f'{pc_acc:.2f}%' for pc_acc in pc_accs[name]]))

    return accs