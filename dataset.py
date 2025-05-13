import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as tvd
from torchvision import transforms,datasets
import numpy as np
import random
from PIL import Image
from collections import Counter

class SPMIDataset(Dataset):
    """Partial‐label dataset wrapper with global‐index return."""
    def __init__(self,
                 dataset_name,
                 root='./data',
                 num_labeled=4000,
                 partial_rate=0.3,
                 transform=None,
                 download=True,
                 seed=42):
        self.dataset_name = dataset_name
        self.num_labeled  = num_labeled
        self.partial_rate = partial_rate
        self.transform    = transform
        self.download     = download

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # 1) load raw arrays + test split
        self._load_dataset(root)
        # 2) split into labeled / unlabeled indices
        self._split_data()
        # 3) generate initial candidate sets for labeled
        self._generate_partial_labels()

    def _load_dataset(self, root):
        if self.dataset_name == 'fashion_mnist':
            ds = tvd.FashionMNIST(root, train=True, download=self.download, transform=None)
            self.data   = ds.data.numpy()
            self.targets= ds.targets.numpy()
            self.num_classes = 10
            self.test_ds = tvd.FashionMNIST(
                root, train=False, download=self.download,
                transform=get_transforms(self.dataset_name, train=False, strong_aug=False)
            )
        elif self.dataset_name == 'cifar10':
            ds = tvd.CIFAR10(root, train=True, download=self.download, transform=None)
            self.data   = ds.data
            self.targets= np.array(ds.targets)
            self.num_classes = 10
            self.test_ds = tvd.CIFAR10(
                root, train=False, download=self.download,
                transform=get_transforms(self.dataset_name, train=False, strong_aug=False)
            )
        elif self.dataset_name == 'cifar100':
            ds = tvd.CIFAR100(root, train=True, download=self.download, transform=None)
            self.data   = ds.data
            self.targets= np.array(ds.targets)
            self.num_classes = 100
            self.test_ds = tvd.CIFAR100(
                root, train=False, download=self.download,
                transform=get_transforms(self.dataset_name, train=False, strong_aug=False)
            )
        elif self.dataset_name == 'svhn':
            ds = tvd.SVHN(root, split='train', download=self.download, transform=None)
            self.data = np.transpose(ds.data, (0, 2, 3, 1))
            self.targets = np.array(ds.labels)
            self.num_classes = 10
            self.test_ds = tvd.SVHN(
                root, split='test', download=self.download,
                transform=get_transforms(self.dataset_name, train=False, strong_aug=False)
            )
        else:
            raise ValueError(f"Unknown dataset {self.dataset_name}")

    def _split_data(self):
        idxs = list(range(len(self.targets)))
        random.shuffle(idxs)
        per_cls = self.num_labeled // self.num_classes

        cls_to_idxs = {c: [] for c in range(self.num_classes)}
        for i in idxs:
            cls_to_idxs[self.targets[i]].append(i)

        self.labeled_indices = []
        for c, lst in cls_to_idxs.items():
            if len(lst) < per_cls:
                raise ValueError(f"Not enough examples of class {c}")
            self.labeled_indices += lst[:per_cls]
        self.unlabeled_indices = [i for i in idxs if i not in self.labeled_indices]

        print(f"{self.dataset_name}: {len(self.labeled_indices)} labeled, "
              f"{len(self.unlabeled_indices)} unlabeled.")

    def _generate_partial_labels(self):
        self.candidate_labels = [[] for _ in range(len(self.targets))]
        for idx in self.labeled_indices:
            true = self.targets[idx]
            choices = [true]
            for c in range(self.num_classes):
                if c != true and random.random() < self.partial_rate:
                    choices.append(c)
            while len(choices) < 2:
                c = random.randrange(self.num_classes)
                if c != true and c not in choices:
                    choices.append(c)
            self.candidate_labels[idx] = choices

        sizes = [len(self.candidate_labels[i]) for i in self.labeled_indices]
        from collections import Counter
        dist = Counter(sizes)
        avg = sum(sizes) / len(sizes)
        print(f"Avg candidates/labeled: {avg:.2f}, dist={dict(dist)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        if self.dataset_name == 'fashion_mnist':
            img = Image.fromarray(img, mode='L')
        else:
            img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        mask = torch.zeros(self.num_classes, dtype=torch.bool)
        for c in self.candidate_labels[idx]:
            mask[c] = True

        is_lab = torch.tensor(idx in self.labeled_indices, dtype=torch.bool)
        target = self.targets[idx]
        return img, mask, target, idx, is_lab

    def update_candidate_labels(self, idx, new_cands):
        if isinstance(new_cands, torch.Tensor):
            new_cands = new_cands.nonzero(as_tuple=True)[0].tolist()
        self.candidate_labels[idx] = new_cands or [self.targets[idx]]

    def get_candidate_masks(self):
        M = torch.zeros(len(self.data), self.num_classes, dtype=torch.bool)
        for i, cands in enumerate(self.candidate_labels):
            for c in cands:
                M[i, c] = True
        return M

    def get_test_loader(self, batch_size=256, num_workers=2):
        return DataLoader(self.test_ds,
                          batch_size=batch_size,
                          shuffle=False,
                          num_workers=num_workers)


# ------ transforms & augmentation helpers ------

class Cutout:
    """Cutout on a tensor image (C×H×W)."""
    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length  = length

    def __call__(self, img):
        # expect img: Tensor[C, H, W]
        c, h, w = img.shape
        mask = torch.ones((h, w), dtype=img.dtype, device=img.device)
        for _ in range(self.n_holes):
            y = random.randrange(h)
            x = random.randrange(w)
            y1, y2 = max(0, y - self.length//2), min(h, y + self.length//2)
            x1, x2 = max(0, x - self.length//2), min(w, x + self.length//2)
            mask[y1:y2, x1:x2] = 0.0
        mask = mask.unsqueeze(0).expand(c, -1, -1)
        return img * mask

class AutoAugment:
    def __init__(self, dataset_name):
        from torchvision.transforms import AutoAugmentPolicy, AutoAugment as TA
        if dataset_name == 'svhn':
            policy = AutoAugmentPolicy.SVHN
        elif dataset_name in ('cifar10', 'cifar100'):
            policy = AutoAugmentPolicy.CIFAR10
        else:
            policy = AutoAugmentPolicy.IMAGENET
        self.aug = TA(policy)

    def __call__(self, img):
        return self.aug(img)

def get_transforms(dataset_name, train=True, strong_aug=False):
    # choose normalization & image size
    if dataset_name == 'fashion_mnist':
        mean, std = (0.5,), (0.5,)
        size, pad = 28, 4
    elif dataset_name in ['cifar10', 'cifar100']:
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        size, pad = 32, 4
    else:  # svhn
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        size, pad = 32, 4

    normalize = transforms.Normalize(mean, std)

    if train:
        tf_list = [
            transforms.RandomCrop(size, padding=pad),
            transforms.RandomHorizontalFlip(),
        ]
        # PIL-based strong augment first
        if strong_aug:
            tf_list.append(AutoAugment(dataset_name))
        # ToTensor + Normalize
        tf_list += [
            transforms.ToTensor(),
            normalize,
        ]
        # tensor-based strong augment last
        if strong_aug:
            tf_list.append(Cutout(n_holes=1, length=16))
    else:
        # Test/eval: no randomness, only tensor & normalize
        tf_list = [
            transforms.ToTensor(),
            normalize,
        ]

    return transforms.Compose(tf_list)