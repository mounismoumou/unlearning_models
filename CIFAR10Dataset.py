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
        elif split == 'val':
            self.dataset = datasets.CIFAR10(root=root, train=True, download=True, transform=self.transform)
            self.dataset.data = self.dataset.data[40000:]  # Last 10,000 as validation
            self.dataset.targets = self.dataset.targets[40000:]
        elif split == 'test':
            self.dataset = datasets.CIFAR10(root=root, train=False, download=True, transform=self.transform)
        else:
            raise ValueError('Unknown split {}'.format(split))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return {'image': image, 'label': label}


def get_dataset(batch_size=64, root=''):
    """Get the CIFAR-10 dataset."""
    
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_ds = CIFAR10Dataset(root=root, split='train', transform=transform)
    val_ds = CIFAR10Dataset(root=root, split='val', transform=transform)
    test_ds = CIFAR10Dataset(root=root, split='test', transform=transform)

    # Get all person id from the training dataset.
    ids = np.array([int(t['label']) for t in train_ds])
    # Create a split that respects the label
    x = np.arange(len(ids)).reshape((-1, 1))

    # Split into retain and forget sets, ensuring separation of labels.
    sklearn.utils.check_random_state(0)
    lpgo = model_selection.LeavePGroupsOut(n_groups=2)
    retain_index, forget_index = next(lpgo.split(x, None, ids))
    retain_ds = data.Subset(train_ds, retain_index)
    forget_ds = data.Subset(train_ds, forget_index)

    if len(set(retain_ds.indices).intersection(set(forget_ds.indices))) > 0:
        raise AssertionError("Overlap in retain and forget sets")

    train_loader = data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    retain_loader = data.DataLoader(retain_ds, batch_size=batch_size, shuffle=True)
    forget_loader = data.DataLoader(forget_ds, batch_size=batch_size, shuffle=True)
    forget_loader_no_shuffle = data.DataLoader(forget_ds, batch_size=batch_size, shuffle=False)

    # Compute class weights
    class_counts = np.bincount([t['label'] for t in train_ds])
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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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