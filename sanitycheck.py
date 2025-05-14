import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
import numpy as np
import random

def main():
    # Settings
    dataset_root = './data'
    num_epochs = 50
    batch_size = 128
    lr = 0.1
    momentum = 0.9
    weight_decay = 5e-4
    subset_fraction = 0.1  # use 10% of training data
    seed = 42

    # Reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    # Load data subset
    full_train = datasets.CIFAR10(dataset_root, train=True, download=True,
                                  transform=transform_train)
    num_samples = int(len(full_train) * subset_fraction)
    indices = list(range(len(full_train)))
    random.shuffle(indices)
    train_subset = Subset(full_train, indices[:num_samples])

    train_loader = DataLoader(train_subset, batch_size=batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    test_set = datasets.CIFAR10(dataset_root, train=False, download=True,
                                transform=transform_test)
    test_loader = DataLoader(test_set, batch_size=batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)

    # Model: ResNet-18 adapted for CIFAR-10
    model = models.resnet18(pretrained=False, num_classes=10)
    # Adjust conv1 & remove maxpool for 32x32 input
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                          weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Training loop
    for epoch in range(1, num_epochs+1):
        model.train()
        running_loss = total = correct = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)

        scheduler.step()
        train_acc = 100. * correct / total

        # Evaluation
        model.eval()
        total = correct = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, preds = outputs.max(1)
                correct += preds.eq(targets).sum().item()
                total += targets.size(0)
        test_acc = 100. * correct / total

        print(f"Epoch {epoch:03d}/{num_epochs}: Train Acc: {train_acc:.2f}%  Test Acc: {test_acc:.2f}%")

    print("Sanity check complete.")

if __name__ == '__main__':
    main()
