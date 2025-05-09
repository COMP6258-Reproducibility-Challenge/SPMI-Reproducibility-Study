import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import random
from PIL import Image


class SPMIDataset:
    #dataset wrapper for spmi added unlabeled data also 
    def __init__(self, dataset_name, root='./data', num_labeled=4000,
                partial_rate=0.3, transform=None, download=True, seed=42):
        self.dataset_name = dataset_name
        self.num_labeled = num_labeled
        self.partial_rate = partial_rate
        self.transform = transform
        self.download = download
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        
        # load the selected dataset
        self._load_dataset(root)
        
        # split into labeled and unlabeled data
        self._split_data()
        
        # Screw up labeled data by generating partial labels
        self._generate_partial_labels()
    
    def _load_dataset(self, root):
        #This downloads and loads the datasets
        if self.dataset_name == 'fashion_mnist':
            dataset = datasets.FashionMNIST(
                root=root, train=True, download=self.download, transform=None
            )
            self.data = dataset.data.numpy()
            self.targets = dataset.targets.numpy()
            self.num_classes = 10
        elif self.dataset_name == 'cifar10':
            dataset = datasets.CIFAR10(
                root=root, train=True, download=self.download, transform=None
            )
            self.data = dataset.data
            self.targets = np.array(dataset.targets)
            self.num_classes = 10
        elif self.dataset_name == 'cifar100':
            dataset = datasets.CIFAR100(
                root=root, train=True, download=self.download, transform=None
            )
            self.data = dataset.data
            self.targets = np.array(dataset.targets)
            self.num_classes = 100
        elif self.dataset_name == 'svhn':
            dataset = datasets.SVHN(
                root=root, split='train', download=self.download, transform=None
            )
            self.data = dataset.data
            self.targets = dataset.labels
            self.num_classes = 10
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
        
        # Create test dataset as well
        if self.dataset_name == 'fashion_mnist':
            self.test_dataset = datasets.FashionMNIST(
                root=root, train=False, download=self.download, transform=get_transforms(self.dataset_name, strong_aug=False)
            )
        elif self.dataset_name == 'cifar10':
            self.test_dataset = datasets.CIFAR10(
                root=root, train=False, download=self.download, transform=get_transforms(self.dataset_name, strong_aug=False)
            )
        elif self.dataset_name == 'cifar100':
            self.test_dataset = datasets.CIFAR100(
                root=root, train=False, download=self.download, transform=get_transforms(self.dataset_name, strong_aug=False)
            )
        elif self.dataset_name == 'svhn':
            self.test_dataset = datasets.SVHN(
                root=root, split='test', download=self.download, transform=get_transforms(self.dataset_name, strong_aug=False)
            )
    
    def _split_data(self):
        # shuffle indices
        indices = list(range(len(self.targets)))
        random.shuffle(indices)
        
        # goup indices by class
        class_indices = [[] for _ in range(self.num_classes)]
        for idx in indices:
            class_indices[self.targets[idx]].append(idx)
        
        # pick equal instances per class for labeled set
        labeled_indices = []
        samples_per_class = self.num_labeled // self.num_classes
        
        for c in range(self.num_classes):
            if len(class_indices[c]) < samples_per_class:
                raise ValueError(f"Not enough samples in class {c}")
            labeled_indices.extend(class_indices[c][:samples_per_class])
        
        # Keep everyting else for unlabeled set
        unlabeled_indices = [idx for idx in indices if idx not in labeled_indices]
        
        self.labeled_indices = labeled_indices
        self.unlabeled_indices = unlabeled_indices
        print(f"Dataset {self.dataset_name}: {len(self.labeled_indices)} labeled instances, {len(self.unlabeled_indices)} unlabeled instances")
    
    def _generate_partial_labels(self):
        #initialize candidate label sets for all data
        self.candidate_labels = [[] for _ in range(len(self.targets))]
        
        # generate partial labels for labeled data
        for idx in self.labeled_indices:
            true_label = self.targets[idx]
            # qlways include the true label
            labels = [true_label]
            
            # based on the p rate add incorrect labels
            max_candidates = max(2, int(self.partial_rate * self.num_classes))
            while len(labels) < max_candidates:
                candidate = random.randint(0, self.num_classes - 1)
                if candidate != true_label and candidate not in labels:
                    labels.append(candidate)
            
            self.candidate_labels[idx] = labels
            
        # average number of candidates
        avg_candidates = sum(len(self.candidate_labels[idx]) for idx in self.labeled_indices) / len(self.labeled_indices)
        print(f"Average number of candidates per labeled instance: {avg_candidates:.2f}")
    
    def __len__(self):
        #total no. of instances
        return len(self.data)
    
    def __getitem__(self, idx):
       #get an instance
        img, target = self.data[idx], self.targets[idx]
        
        # image transforms
        if self.dataset_name == 'fashion_mnist':
            img = Image.fromarray(img, mode='L')
        elif self.dataset_name == 'svhn':
            img = np.transpose(img, (1, 2, 0))
            img = Image.fromarray(img)
        else:  # CIFAR-10 andCIFAR-100
            img = Image.fromarray(img)
        
        if self.transform:
            img = self.transform(img)
        
        # Create candidate label mask
        candidate_mask = torch.zeros(self.num_classes)
        for label in self.candidate_labels[idx]:
            candidate_mask[label] = 1
        
        # Is this a labeled instance?
        is_labeled = torch.tensor(idx in self.labeled_indices, dtype=torch.int)
        
        return img, candidate_mask, target, idx, is_labeled
    
    def update_candidate_labels(self, idx, new_candidates):
        #candidate label update
        if isinstance(new_candidates, torch.Tensor):
            # convert from mask to list of indices
            new_candidates = new_candidates.nonzero(as_tuple=True)[0].tolist()
        self.candidate_labels[idx] = new_candidates
        
        # should ensure at least one candidate is there
        if len(self.candidate_labels[idx]) == 0:
            # if we have nothing then add the true label
            self.candidate_labels[idx] = [self.targets[idx]]
    
    def get_candidate_masks(self):
        masks = torch.zeros(len(self.data), self.num_classes)
        for idx, candidates in enumerate(self.candidate_labels):
            for label in candidates:
                masks[idx, label] = 1
        return masks
        
    def get_test_loader(self, batch_size=256):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )


# HELPER FUNC : ALL THE TRANSFORMS ARE HERE
def get_transforms(dataset_name, strong_aug=False):
    if dataset_name == 'fashion_mnist':
        # weak aug
        weak_transform = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        
        # No AutoAugment for F-MNIST as per paper
        if strong_aug:
            strong_transform = transforms.Compose([
                transforms.RandomCrop(28, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                Cutout(n_holes=1, length=16)
            ])
        else:
            strong_transform = weak_transform
    else:  # CIFAR-10,CIFAR-100,SVHN
        weak_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        
        if strong_aug:
            strong_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                AutoAugment(dataset_name),
                transforms.ToTensor(),
                Cutout(n_holes=1, length=16)
            ])
        else:
            strong_transform = weak_transform
    
    return weak_transform if not strong_aug else strong_transform


class Cutout:
    #cutout implementation
    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length
    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)
        mask = np.ones((h, w), np.float32)
        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        return img
class AutoAugment:
    #Autoaug
    def __init__(self, dataset_name):
        from torchvision.transforms import AutoAugment as TvAutoAugment
        from torchvision.transforms import AutoAugmentPolicy
        if dataset_name == 'svhn':
            self.policy = TvAutoAugment(AutoAugmentPolicy.SVHN)
        elif dataset_name in ['cifar10', 'cifar100']:
            self.policy = TvAutoAugment(AutoAugmentPolicy.CIFAR10)
        else:
            self.policy = TvAutoAugment(AutoAugmentPolicy.IMAGENET)
    def __call__(self, img):
        return self.policy(img)