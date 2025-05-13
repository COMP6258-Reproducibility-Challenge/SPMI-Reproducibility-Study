# ablation_study.py

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import random
import json
import csv
from datetime import datetime

from dataset import SPMIDataset, get_transforms
from model import get_model
from spmi import SPMI
from train import create_ema_model, update_ema_model, evaluate

# Set seeds for reproducibility
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class AblationSPMI(SPMI):
    """SPMI with toggles for ablation components"""
    def __init__(self, model, num_classes, tau, unlabeled_tau,
                 use_init, use_LG, use_LC, **kwargs):
        super().__init__(model, num_classes, tau, unlabeled_tau, **kwargs)
        self.use_init = use_init
        self.use_LG = use_LG
        self.use_LC = use_LC

    def initialize_unlabeled(self, loader, device):
        if self.use_init:
            return super().initialize_unlabeled(loader, device)
        else:
            print("[Ablation] Skipping init")

    def expand_labels(self, *args, **kwargs):
        if self.use_LG:
            return super().expand_labels(*args, **kwargs)
        else:
            print("[Ablation] Skipping label generation")
            return args[1]  # return unchanged masks

    def condense_labels(self, *args, **kwargs):
        if self.use_LC:
            return super().condense_labels(*args, **kwargs)
        else:
            print("[Ablation] Skipping label condensation")
            return args[1]


def run_ablation(config, ablation_flags, save_dir):
    set_seeds(config['seed'])
    device = torch.device(config['device'])
    # Unpack flags
    use_init, use_LG, use_LC = ablation_flags
    exp_name = []
    if use_init: exp_name.append('init')
    if use_LG:   exp_name.append('LG')
    if use_LC:   exp_name.append('LC')
    exp_name = "_".join(exp_name) or 'none'
    print(f"Starting {exp_name} on {config['dataset']}")

    # Data
    train_tf = get_transforms(config['dataset'], train=True, strong_aug=True)
    dataset = SPMIDataset(config['dataset'], './data',
                          config['num_labeled'], config['partial_rate'],
                          transform=train_tf, seed=config['seed'])
    train_loader = DataLoader(dataset, batch_size=config['batch_size'],
                              shuffle=True, num_workers=config['num_workers'],
                              pin_memory=True)
    test_loader = dataset.get_test_loader(batch_size=config['batch_size'],
                                          num_workers=config['num_workers'])

    # Model - FIX: Move to device!
    model = get_model(config['model_name'], dataset.num_classes, in_channels=config['in_ch'])
    model = model.to(device)  # This was missing!
    
    spmi = AblationSPMI(model, dataset.num_classes,
                        tau=config['tau'], unlabeled_tau=config['unlabeled_tau'],
                        use_init=use_init, use_LG=use_LG, use_LC=use_LC)
    optimizer = optim.SGD(model.parameters(), config['lr'], momentum=config['momentum'],
                          weight_decay=config['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'])
    ema_model = create_ema_model(model) if config['use_ema'] else None
    original_masks = dataset.get_candidate_masks().to(device)

    # Metrics
    results = []
    for epoch in range(config['epochs']):
        loss = spmi.train_epoch(train_loader, optimizer, device,
                                epoch, config['warmup'], original_masks)
        scheduler.step()
        if ema_model: update_ema_model(model, ema_model, config['ema_decay'])
        eval_model = ema_model or model
        acc = evaluate(eval_model, test_loader, device)
        results.append((epoch, acc))
        if epoch % config['print_interval'] == 0:
            print(f"Epoch {epoch}/{config['epochs']} Acc: {acc:.2f}%")

    # Save summary
    summary = os.path.join(save_dir, f"{config['dataset']}_{exp_name}.json")
    with open(summary, 'w') as f:
        json.dump({'exp':exp_name, 'dataset':config['dataset'], 'results':results}, f)
    print(f"Saved to {summary}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['cifar10','fmnist','cifar100','svhn'],
                        required=True)
    args = parser.parse_args()

    # Base config
    cfg = {
        'dataset': args.dataset,
        'num_labeled': 4000,
        'partial_rate': 0.3,
        'epochs': 500,
        'warmup': 10,
        'batch_size': 256,
        'lr': 0.03,
        'momentum': 0.9,
        'weight_decay':5e-4,
        'tau':3.0,
        'unlabeled_tau':2.0,
        'use_ema':False,
        'ema_decay':0.999,
        'device':'cuda' if torch.cuda.is_available() else 'cpu',
        'num_workers':4,
        'seed':42,
        'model_name': 'wrn-28-2',
        'in_ch': 1 if args.dataset=='fmnist' else 3,
        'print_interval':10
    }

    save_dir = f"ablation_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(save_dir, exist_ok=True)

    # Only run full SPMI model
    flags = (True, True, True)
    run_ablation(cfg, flags, save_dir)

if __name__ == '__main__':
    main()