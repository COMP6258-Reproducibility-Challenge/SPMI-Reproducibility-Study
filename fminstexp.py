# fmnist_exp.py

import copy
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets
from dataset import SPMIDataset, get_transforms
from model import get_model
from spmi import SPMI
from train import update_ema_model

def create_ema_model(model):
    ema = copy.deepcopy(model)
    for p in ema.parameters():
        p.requires_grad_(False)
    return ema

def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, targets in loader:
            imgs, targets = imgs.to(device), targets.to(device)
            pred = model(imgs).argmax(1)
            correct += (pred == targets).sum().item()
            total   += targets.size(0)
    return 100 * correct / total

def main():
    # Hyperparameters from the paper for Fashion-MNIST
    dataset_name   = 'fashion_mnist'
    num_labeled    = 1000  # Paper uses 1000 and 4000, starting with 1000
    partial_rate   = 0.3   # Paper uses 0.3 for F-MNIST
    warmup_epochs  = 10    # Paper uses warm-up period
    num_epochs     = 500   # Paper uses 500 epochs
    batch_size     = 256   # Paper uses batch size 256
    lr             = 0.03  # Paper uses learning rate 0.03
    weight_decay   = 5e-4  # Paper uses weight decay 5e-4

    # SPMI specific parameters from paper
    tau            = 3.0   # Paper uses τ = 3 for partial label data
    unlabeled_tau  = 2.0   # Paper uses τ = 2 for unlabeled data
    init_threshold = None  # Defaults to 1/C in SPMI
    prior_alpha    = 1.0  # Paper mentions no EMA smoothing for 1.0
    use_ib_penalty = False # Paper doesn't use IB penalty for their main experiments
    ib_beta        = 0.0

    # EMA model - paper doesn't use EMA for inference
    use_ema        = False
    ema_decay      = 0.999

    # Check for available GPUs
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        n_gpus = torch.cuda.device_count()
        print(f"Found {n_gpus} GPU(s)")
        if n_gpus >= 2:
            print("Using multi-GPU setup")
            use_multi_gpu = True
            # Increase batch size proportionally for multi GPU
            batch_size = batch_size * n_gpus
            print(f"Adjusted batch size to {batch_size} for {n_gpus} GPUs")
        else:
            print("Only 1 GPU available, using single GPU")
            use_multi_gpu = False
    else:
        device = torch.device('cpu')
        use_multi_gpu = False
        print("CUDA not available, using CPU")
    
    print(f"Running on {device}\n")

    # Data - Fashion-MNIST specific transforms (no AutoAugment for F-MNIST per appendix)
    transform_train = get_transforms(dataset_name, strong_aug=False)
    train_dataset = SPMIDataset(
        dataset_name,
        './data',
        num_labeled=num_labeled,
        partial_rate=partial_rate,
        transform=transform_train,
        download=True,
        seed=42
    )
    num_workers = min(8 * (n_gpus if use_multi_gpu else 1), 16)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2,
        drop_last=True
    )
    test_transform = get_transforms(dataset_name, train=False, strong_aug=False)
    test_dataset = datasets.FashionMNIST(
        './data', train=False, download=True, transform=test_transform
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0)
    )

    unlabeled_idx = train_dataset.unlabeled_indices

    # Model - Paper uses LeNet for Fashion-MNIST
    model = get_model(
        name='lenet',
        num_classes=train_dataset.num_classes,
        in_channels=1  # Fashion-MNIST is grayscale (1 channel)
    ).to(device)

    # Apply DataParallel for multi GPU
    if use_multi_gpu:
        model = nn.DataParallel(model)
        print(f"Model wrapped with DataParallel across {n_gpus} GPUs")

    spmi = SPMI(
        model=model,
        num_classes=train_dataset.num_classes,
        tau=tau,
        unlabeled_tau=unlabeled_tau,
        init_threshold=init_threshold,
        prior_alpha=prior_alpha,
        use_ib_penalty=use_ib_penalty,
        ib_beta=ib_beta
    )

    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=weight_decay
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=0
    )

    ema_model = create_ema_model(model) if use_ema else None
    original_masks = train_dataset.get_candidate_masks().to(device)

    # Diagnostics storage
    diagnostics = []

    # Training Loop
    for epoch in range(1, num_epochs + 1):
        loss = spmi.train_epoch(
            train_loader,
            optimizer,
            device,
            epoch - 1,
            warmup_epochs,
            original_masks
        )
        scheduler.step()

        if use_ema:
            update_ema_model(model, ema_model, ema_decay)

        eval_model = ema_model or model
        acc = evaluate(eval_model, test_loader, device)

        # Minimal terminal output
        u_masks = train_dataset.get_candidate_masks()[unlabeled_idx]
        avg_u = u_masks.sum(dim=1).float().mean().item()
        print(f"[Epoch {epoch:03d}/{num_epochs}] "
              f"Loss: {loss:.4f}  Acc: {acc:.2f}%  "
              f"AvgUnlabCands: {avg_u:.2f}")

        # Record diagnostics
        # Handle DataParallel case for accessing class_priors
        if use_multi_gpu:
            priors = spmi.class_priors.cpu().tolist()
        else:
            priors = spmi.class_priors.cpu().tolist()
        
        row = {
            'epoch': epoch,
            'loss': loss,
            'test_acc': acc,
            'avg_unlabeled_cands': avg_u
        }
        # Include class priors
        for i, p in enumerate(priors):
            row[f'prior_{i}'] = p
        diagnostics.append(row)

    # Save diagnostics
    keys = diagnostics[0].keys()
    output_file = f'fmnist_paper_experiment_diagnostics_l{num_labeled}_p{partial_rate}_{"multi" if use_multi_gpu else "single"}gpu.csv'
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(diagnostics)

    print(f"\nDiagnostics saved to {output_file}")
    print("Experiment complete")
    print(f"Final accuracy: {acc:.2f}%")

if __name__ == '__main__':
    main()