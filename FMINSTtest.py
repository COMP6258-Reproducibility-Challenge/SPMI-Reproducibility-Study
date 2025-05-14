# Fixed FMINSTtest.py

import copy
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import SPMIDataset, get_transforms
from model import get_model
from spmi import SPMI
from train import update_ema_model  # we will override create_ema_model here
import csv

#Create an EMA model by deep copying the original and freezing its parameters
def create_ema_model(model):
    ema = copy.deepcopy(model)
    for p in ema.parameters():
        p.requires_grad_(False)
    return ema

# Evaluate model on test loader ; Handles both 2 tuple and 5 tuple formats 
# Returns overall accuracy and perclass accuracies
def evaluate_test(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    # Determine number of classes from model output
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 2:
                imgs, targets = batch
            else:
                imgs, _, targets, _, _ = batch
            imgs = imgs.to(device)
            outputs = model(imgs)
            num_classes = outputs.size(1)
            break

    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 2:
                imgs, targets = batch
            else:
                imgs, _, targets, _, _ = batch

            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs, dim=1)

            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            for t, p in zip(targets, predicted):
                class_total[t.item()] += 1
                if p.item() == t.item():
                    class_correct[t.item()] += 1

    overall_acc = 100.0 * correct / total
    per_class_acc = {
        f'class_{i}_acc': (100.0 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0)
        for i in range(num_classes)
    }
    return overall_acc, per_class_acc


if __name__ == '__main__':
    # EXP settings
    dataset_name = 'fashion_mnist'
    num_labeled = 1000
    partial_rate = 0.3
    num_epochs = 20
    warmup_epochs = 5
    batch_size = 128
    learning_rate = 0.03
    weight_decay = 5e-4
    tau = 3.0
    unlabeled_tau = 2.0
    use_ema = True
    ema_decay = 0.999

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Prepare data 
    transform_train = get_transforms(dataset_name, strong_aug=True)
    train_dataset = SPMIDataset(
        dataset_name,
        root='./data',
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
        num_workers=2,
        pin_memory=True
    )
    test_loader = train_dataset.get_test_loader(batch_size=batch_size, num_workers=2)

    # Model setup using factory function 
    model = get_model(
        name='lenet',
        num_classes=train_dataset.num_classes,
        in_channels=1
    ).to(device)
    print(f"Model: LeNet with {train_dataset.num_classes} classes")

    # SPMI setup 
    spmi = SPMI(
        model=model,
        num_classes=train_dataset.num_classes,
        tau=tau,
        unlabeled_tau=unlabeled_tau,
        use_ib_penalty=False,
        ib_beta=0.01
    )

    # Optimizer and scheduler
    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)

    # EMA model
    ema_model = create_ema_model(model) if use_ema else None

    # Diagnostics setup
    diagnostics = []

    # Initial diagnostics
    print(f"\n INITIAL STATE")
    masks = train_dataset.get_candidate_masks()
    labeled_idx = train_dataset.labeled_indices
    unlabeled_idx = train_dataset.unlabeled_indices

    labeled_masks = masks[labeled_idx]
    unlabeled_masks = masks[unlabeled_idx]

    print(f"Dataset statistics:")
    print(f"  Total samples: {len(train_dataset)}")
    print(f"  Labeled samples: {len(labeled_idx)}")
    print(f"  Unlabeled samples: {len(unlabeled_idx)}")
    print(f"  Labeled avg candidates: {labeled_masks.sum(dim=1).float().mean():.2f}")
    print(f"  Unlabeled avg candidates: {unlabeled_masks.sum(dim=1).float().mean():.2f}")
    print(f"Initial class priors: {spmi.class_priors.cpu().numpy()}")

    # Capture original masks
    original_masks = train_dataset.get_candidate_masks().to(device)

    # Training loop
    for epoch in range(num_epochs):
        print(f"\n{'='*50}")
        print(f"EPOCH {epoch+1}/{num_epochs}")
        print(f"{'='*50}")

        # Train one epoch positional args for original_masks
        train_loss = spmi.train_epoch(
            train_loader,
            optimizer,
            device,
            epoch,
            warmup_epochs,
            original_masks
        )

        # Update learning rate
        scheduler.step()

        # Update EMA
        if use_ema:
            update_ema_model(model, ema_model, ema_decay)

        # Evaluate
        eval_model = ema_model if (use_ema and ema_model is not None) else model
        overall_acc, per_class_acc = evaluate_test(eval_model, test_loader, device)

        # Print results
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Test Accuracy: {overall_acc:.2f}%")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

        # Detailed diagnostics for first few epochs
        if epoch < 5 or epoch == warmup_epochs:
            print(f"\n Detailed Diagnostics ")
            current_masks = train_dataset.get_candidate_masks()
            labeled_masks = current_masks[labeled_idx]
            unlabeled_masks = current_masks[unlabeled_idx]

            print(f"  Labeled avg candidates: {labeled_masks.sum(dim=1).float().mean():.2f}")
            print(f"  Unlabeled avg candidates: {unlabeled_masks.sum(dim=1).float().mean():.2f}")

            no_cand_labeled = (labeled_masks.sum(dim=1) == 0).sum().item()
            no_cand_unlabeled = (unlabeled_masks.sum(dim=1) == 0).sum().item()
            all_cand_labeled = (labeled_masks.sum(dim=1) == train_dataset.num_classes).sum().item()
            all_cand_unlabeled = (unlabeled_masks.sum(dim=1) == train_dataset.num_classes).sum().item()

            print(f"Edge cases _> No candidates L/U: {no_cand_labeled}/{no_cand_unlabeled}; "
                  f"All candidates L/U: {all_cand_labeled}/{all_cand_unlabeled}")
            print(f"  Class priors: {[f'{p:.3f}' for p in spmi.class_priors.cpu().tolist()]}")

        # Collect diagnostics
        masks = train_dataset.get_candidate_masks()
        avg_cand_all = masks.sum(dim=1).float().mean().item()
        avg_cand_labeled = masks[labeled_idx].sum(dim=1).float().mean().item()
        avg_cand_unlabeled = masks[unlabeled_idx].sum(dim=1).float().mean().item()
        priors = spmi.class_priors.cpu().tolist()

        row = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'overall_acc': overall_acc,
            'avg_cand_all': avg_cand_all,
            'avg_cand_labeled': avg_cand_labeled,
            'avg_cand_unlabeled': avg_cand_unlabeled,
        }
        row.update(per_class_acc)
        for k in range(train_dataset.num_classes):
            row[f'prior_{k}'] = priors[k]
        diagnostics.append(row)

    # Save results 
    print(f"\n EXPERIMENT COMPLETED ")
    print(f"Final Test Accuracy: {overall_acc:.2f}%")

    csv_file = f'spmi_diagnostics_{dataset_name}_l{num_labeled}_p{partial_rate}.csv'
    fieldnames = [
        'epoch', 'train_loss', 'overall_acc',
        'avg_cand_all', 'avg_cand_labeled', 'avg_cand_unlabeled'
    ] + [f'class_{i}_acc' for i in range(train_dataset.num_classes)] \
      + [f'prior_{i}' for i in range(train_dataset.num_classes)]

    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for data in diagnostics:
            writer.writerow(data)

    print(f"Diagnostics saved to {csv_file}")

