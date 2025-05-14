# svhn_exp.py

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

def run_experiment(num_labeled, partial_rate, seed=42):
    # Hyperparameters frm the paper
    dataset_name   = 'svhn'
    warmup_epochs  = 10
    num_epochs     = 500
    batch_size     = 256
    lr             = 0.03
    weight_decay   = 5e-4

    # SPMI specific
    tau            = 3.0
    unlabeled_tau  = 2.0
    init_threshold = None
    # EMA smoothing
    prior_alpha    = 0.9   
    use_ib_penalty = True
    ib_beta        = 0.01

    # Device Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")

    # Data
    transform_train = get_transforms(dataset_name, strong_aug=True)
    train_dataset = SPMIDataset(
        dataset_name,
        './data',
        num_labeled=num_labeled,
        partial_rate=partial_rate,
        transform=transform_train,
        download=True,
        seed=seed
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    test_transform = get_transforms(dataset_name, train=False, strong_aug=False)
    test_dataset = datasets.SVHN(
        './data', split='test', download=True, transform=test_transform
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Model amd SPMI
    model = get_model(
        name='wrn-28-2',
        num_classes=train_dataset.num_classes,
        in_channels=3
    ).to(device)

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

    original_masks = train_dataset.get_candidate_masks().to(device)
    
    # Training Loop
    diagnostics = []
    best_acc = 0.0
    
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

        acc = evaluate(model, test_loader, device)
        best_acc = max(best_acc, acc)

        # Progress monitoring
        if epoch % 50 == 0 or epoch == num_epochs:
            unlabeled_idx = train_dataset.unlabeled_indices
            u_masks = train_dataset.get_candidate_masks()[unlabeled_idx]
            avg_u = u_masks.sum(dim=1).float().mean().item()
            
            print(f"[Epoch {epoch:03d}/{num_epochs}] "
                  f"Loss: {loss:.4f}  Acc: {acc:.2f}%  "
                  f"Best: {best_acc:.2f}%  AvgUnlabCands: {avg_u:.2f}")

        # Record diagnostics
        priors = spmi.class_priors.cpu().tolist()
        row = {
            'epoch': epoch,
            'loss': loss,
            'test_acc': acc,
            'best_acc': best_acc
        }
        for i, p in enumerate(priors):
            row[f'prior_{i}'] = p
        diagnostics.append(row)

    # Save the Results
    output_file = f'svhn_l{num_labeled}_p{partial_rate}_s{seed}.csv'
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=diagnostics[0].keys())
        writer.writeheader()
        writer.writerows(diagnostics)

    print(f"Results saved to {output_file}")
    return best_acc

def main():
    print("Running SVHN experiments from the paper")
    
    # Paper experiments
    experiments = [
        (1000, 0.3),  # Table 1
    ]
    
    results = {}
    for num_labeled, partial_rate in experiments:
        print(f"\n{'='*50}")
        print(f"Experiment: l={num_labeled}, p={partial_rate}")
        print(f"{'='*50}")
        
        # Run with different seeds for statistical significance
        accs = []
        for seed in [42]:
            print(f"\nRunning with seed {seed}...")
            acc = run_experiment(num_labeled, partial_rate, seed)
            accs.append(acc)
        
        mean_acc = sum(accs) / len(accs)
        std_acc = (sum((x - mean_acc)**2 for x in accs) / len(accs)) ** 0.5
        
        results[(num_labeled, partial_rate)] = (mean_acc, std_acc)
        print(f"\nResults for l={num_labeled}, p={partial_rate}:")
        print(f"Mean Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY - SVHN Results")
    print(f"{'='*60}")
    for (num_labeled, partial_rate), (mean_acc, std_acc) in results.items():
        print(f"l={num_labeled}, p={partial_rate}: {mean_acc:.2f}% ± {std_acc:.2f}%")

if __name__ == '__main__':
    main()