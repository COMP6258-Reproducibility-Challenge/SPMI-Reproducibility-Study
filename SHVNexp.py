# svhnexp.py

import copy
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

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
    # ─── Hyperparameters (paper settings) ────────────────────────
    dataset_name   = 'svhn'
    num_labeled    = 1000
    partial_rate   = 0.3
    warmup_epochs  = 10
    num_epochs     = 500
    batch_size     = 256
    lr             = 0.03
    weight_decay   = 5e-4

    # SPMI-specific
    tau            = 3.0
    unlabeled_tau  = 2.0
    init_threshold = None  # defaults to 1/C in SPMI
    prior_alpha    = 1.0   # no EMA smoothing
    use_ib_penalty = False
    ib_beta        = 0.0

    # EMA model toggle (paper does not use EMA for inference)
    use_ema        = False
    ema_decay      = 0.999

    # ─── Multi-GPU Setup ─────────────────────────────────────────
    # Check for available GPUs
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        n_gpus = torch.cuda.device_count()
        print(f"Found {n_gpus} GPU(s)")
        if n_gpus >= 2:
            print("Using multi-GPU setup")
            use_multi_gpu = True
            # Increase batch size proportionally for multi-GPU
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

    # ─── Data ────────────────────────────────────────────────────
    transform_train = get_transforms(dataset_name, strong_aug=True)
    train_dataset = SPMIDataset(
        dataset_name,
        './data',
        num_labeled=num_labeled,
        partial_rate=partial_rate,
        transform=transform_train,
        download=True,
        seed=42
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4 * (n_gpus if use_multi_gpu else 1),  # Scale workers with GPUs
        pin_memory=True
    )
    test_loader = train_dataset.get_test_loader(
        batch_size=batch_size,
        num_workers=4 * (n_gpus if use_multi_gpu else 1),
    )

    unlabeled_idx = train_dataset.unlabeled_indices

    # ─── Model & SPMI ────────────────────────────────────────────
    model = get_model(
        name='wrn-28-2',
        num_classes=train_dataset.num_classes,
        in_channels=3
    ).to(device)

    # ─── Apply DataParallel for multi-GPU ────────────────────────
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

    # ─── Diagnostics storage ────────────────────────────────────
    diagnostics = []

    # ─── Training Loop ─────────────────────────────────────────
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
        priors = spmi.class_priors.cpu().tolist()
        row = {
            'epoch': epoch,
            'loss': loss,
            'test_acc': acc,
            'avg_unlabeled_cands': avg_u
        }
        # include class priors
        for i, p in enumerate(priors):
            row[f'prior_{i}'] = p
        diagnostics.append(row)

    # ─── Save diagnostics ────────────────────────────────────────
    keys = diagnostics[0].keys()
    output_file = f'svhn_paper_experiment_diagnostics_{"multi" if use_multi_gpu else "single"}gpu.csv'
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(diagnostics)

    print(f"\n▶ Diagnostics saved to {output_file}")
    print("▶ Paper experiment complete!")

if __name__ == '__main__':
    main()