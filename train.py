# train.py

import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import SPMIDataset, get_transforms
from model import get_model
from spmi import SPMI
import copy


def create_ema_model(model):
    ema = copy.deepcopy(model)
    for p in ema.parameters():
        p.requires_grad_(False)
    return ema


def update_ema_model(model, ema_model, decay):
    with torch.no_grad():
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.mul_(decay).add_(p, alpha=1 - decay)


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            if len(data) == 2:
                imgs, targets = data
            else:
                imgs, _, targets, _, _ = data
            
            imgs = imgs.to(device)
            targets = targets.to(device)
            outs = model(imgs)
            pred = outs.argmax(dim=1)
            correct += (pred == targets).sum().item()
            total += targets.size(0)
    return 100 * correct / total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',    type=str,   default='cifar10',
                        choices=['fashion_mnist','cifar10','cifar100','svhn'])
    parser.add_argument('--root',       type=str,   default='./data')
    parser.add_argument('--batch_size', type=int,   default=256)
    parser.add_argument('--num_labeled',type=int,   default=4000)
    parser.add_argument('--partial_rate', type=float, default=0.3)
    parser.add_argument('--epochs',     type=int,   default=500)
    parser.add_argument('--warmup',     type=int,   default=10)
    parser.add_argument('--lr',         type=float, default=0.03)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum',   type=float, default=0.9)
    parser.add_argument('--tau',        type=float, default=3.0)
    parser.add_argument('--unlabeled_tau', type=float, default=2.0)
    parser.add_argument('--use_ema',    action='store_true')
    parser.add_argument('--ema_decay',  type=float, default=0.999)
    parser.add_argument('--eval_interval', type=int, default=10)
    parser.add_argument('--device',     type=str,   default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # --- Transforms & DataLoaders ---
    train_tf = get_transforms(args.dataset, strong_aug=True)
    test_tf  = get_transforms(args.dataset, strong_aug=False)

    dataset = SPMIDataset(
        dataset_name=args.dataset,
        root=args.root,
        num_labeled=args.num_labeled,
        partial_rate=args.partial_rate,
        transform=train_tf,
        download=True,
        seed=42
    )
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    test_loader = dataset.get_test_loader(
        batch_size=args.batch_size,
        num_workers=4
    )

    # --- Model & SPMI setup ---
    if args.dataset == 'fashion_mnist':
        model_name, in_ch = 'lenet', 1
    elif args.dataset in ['cifar10', 'svhn']:
        model_name, in_ch = 'wrn-28-2', 3
    else:
        model_name, in_ch = 'wrn-28-8', 3

    model = get_model(
        name=model_name,
        num_classes=dataset.num_classes,
        in_channels=in_ch
    ).to(args.device)

    spmi = SPMI(
        model=model,
        num_classes=dataset.num_classes,
        tau=args.tau,
        unlabeled_tau=args.unlabeled_tau
    )

    optimizer = SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)

    ema_model = create_ema_model(model) if args.use_ema else None

    # capture original (labeled) masks once
    original_masks = dataset.get_candidate_masks().to(args.device)

    # --- Training Loop ---
    for epoch in range(args.epochs):
        loss = spmi.train_epoch(
            dataloader=train_loader,
            optimizer=optimizer,
            device=args.device,
            epoch=epoch,
            warmup_epochs=args.warmup,
            original_masks=original_masks
        )
        scheduler.step()

        if args.use_ema:
            update_ema_model(model, ema_model, args.ema_decay)

        if epoch % args.eval_interval == 0 or epoch == args.epochs-1:
            eval_model = ema_model if ema_model is not None else model
            acc = evaluate(eval_model, test_loader, args.device)
            print(f"[Epoch {epoch:03d}/{args.epochs}] "
                  f"Loss: {loss:.4f}  Acc: {acc:.2f}%")

if __name__ == '__main__':
    main()
