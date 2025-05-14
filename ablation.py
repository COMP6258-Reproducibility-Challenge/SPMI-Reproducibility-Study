import argparse
import os
import torch

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

# set seeds
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# Class to toggle SMPI with ablation steps
class AblationSPMI(SPMI):
    def __init__(self, model, num_classes, tau, unlabeled_tau,
                 use_init, use_LG, use_LC, **kwargs):
        super().__init__(model, num_classes, tau, unlabeled_tau, **kwargs)
        self.use_init = use_init
        self.use_LG = use_LG
        self.use_LC = use_LC

    def initialize_unlabeled(self, dataloader, device):
        if self.use_init:
            return super().initialize_unlabeled(dataloader, device)
        else:
            print("Ablation Skipping initialization")
            # Return the current candidate masks without modification
            # This ensures unlabeled data keeps whatever candidate sets it has
            return dataloader.dataset.get_candidate_masks().to(device)

    def expand_labels(self, outputs, candidate_masks, is_labeled, indices, original_masks=None):
        if self.use_LG:
            return super().expand_labels(outputs, candidate_masks, is_labeled, indices, original_masks)
        else:
            print("Ablation Skipping label generation")
            return candidate_masks  # Return unchanged masks

    def condense_labels(self, outputs, candidate_masks, is_labeled):
        if self.use_LC:
            return super().condense_labels(outputs, candidate_masks, is_labeled)
        else:
            print("Ablation Skipping label condensation")
            return candidate_masks  # Return unchanged masks

#Get dataset specific config
def get_dataset_config(dataset_name):
    # Dataset name mapping
    dataset_mapping = {
        'fmnist': 'fashion_mnist',
        'cifar10': 'cifar10',
        'cifar100': 'cifar100',
        'svhn': 'svhn'
    }
    # Dataset specific parameters based on paper Tables 1 & 2 from the paper
    if dataset_name == 'fmnist':
        return {
            'dataset': dataset_mapping[dataset_name],
            'model_name': 'lenet',
            'in_ch': 1,
            'configs': [
                {'num_labeled': 1000, 'partial_rate': 0.3},
                {'num_labeled': 4000, 'partial_rate': 0.3}
            ]
        }
    elif dataset_name == 'cifar10':
        return {
            'dataset': dataset_mapping[dataset_name],
            'model_name': 'wrn-28-2',
            'in_ch': 3,
            'configs': [
                {'num_labeled': 1000, 'partial_rate': 0.3},
                {'num_labeled': 4000, 'partial_rate': 0.3}
            ]
        }
    elif dataset_name == 'cifar100':
        return {
            'dataset': dataset_mapping[dataset_name],
            'model_name': 'wrn-28-8',
            'in_ch': 3,
            'configs': [
                {'num_labeled': 5000, 'partial_rate': 0.05},
                {'num_labeled': 10000, 'partial_rate': 0.05}
            ]
        }
    elif dataset_name == 'svhn':
        return {
            'dataset': dataset_mapping[dataset_name],
            'model_name': 'wrn-28-2',
            'in_ch': 3,
            'configs': [
                {'num_labeled': 1000, 'partial_rate': 0.3}
            ]
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

# Run the ablation
def run_ablation(config, ablation_flags, save_dir):
    set_seeds(config['seed'])
    device = torch.device(config['device'])
    
    # Unpack flags and create experiment name
    use_init, use_LG, use_LC = ablation_flags
    exp_name = []
    if use_init: exp_name.append('init')
    if use_LG:   exp_name.append('LG')
    if use_LC:   exp_name.append('LC')
    exp_name = "_".join(exp_name) if exp_name else 'none'
    
    exp_full_name = f"{config['dataset']}_l{config['num_labeled']}_p{config['partial_rate']}_{exp_name}"
    print(f"Starting {exp_full_name}")

    # Data
    train_tf = get_transforms(config['dataset'], train=True, strong_aug=True)
    dataset = SPMIDataset(
        config['dataset'], './data',
        config['num_labeled'], config['partial_rate'],
        transform=train_tf, seed=config['seed'], download=True
    )
    
    train_loader = DataLoader(
        dataset, 
        batch_size=config['batch_size'],
        shuffle=True, 
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    test_loader = dataset.get_test_loader(
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )

    # Model
    model = get_model(config['model_name'], dataset.num_classes, in_channels=config['in_ch'])
    model = model.to(device)
    
    # SPMI with ablation
    spmi = AblationSPMI(
        model, dataset.num_classes,
        tau=config['tau'], unlabeled_tau=config['unlabeled_tau'],
        use_init=use_init, use_LG=use_LG, use_LC=use_LC
    )
    
    # Optimizer and scheduler
    optimizer = optim.SGD(
        model.parameters(), 
        config['lr'], 
        momentum=config['momentum'],
        weight_decay=config['weight_decay']
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'])
    
    # EMA model
    ema_model = create_ema_model(model) if config['use_ema'] else None
    
    # Original masks
    original_masks = dataset.get_candidate_masks().to(device)

    # Training loop
    results = []
    for epoch in range(config['epochs']):
        loss = spmi.train_epoch(
            train_loader, optimizer, device,
            epoch, config['warmup'], original_masks
        )
        scheduler.step()
        
        if ema_model: 
            update_ema_model(model, ema_model, config['ema_decay'])
        
        eval_model = ema_model or model
        acc = evaluate(eval_model, test_loader, device)
        results.append({'epoch': epoch, 'loss': loss, 'accuracy': acc})
        
        if epoch % config['print_interval'] == 0 or epoch == config['epochs'] - 1:
            print(f"  Epoch {epoch:3d}/{config['epochs']} Loss: {loss:.4f} Acc: {acc:.2f}%")

    # Save the results
    result_file = os.path.join(save_dir, f"{exp_full_name}_results.json")
    with open(result_file, 'w') as f:
        json.dump({
            'experiment': exp_full_name,
            'config': config,
            'ablation_flags': {'use_init': use_init, 'use_LG': use_LG, 'use_LC': use_LC},
            'final_accuracy': results[-1]['accuracy'],
            'results': results
        }, f, indent=2)
    
    print(f"  Final accuracy: {results[-1]['accuracy']:.2f}%")
    print(f"  Saved to {result_file}")
    
    return results[-1]['accuracy']

# run a full ablation study
def run_full_ablation_study(args):
    
    # Get dataset configuration
    dataset_config = get_dataset_config(args.dataset)
    
    # Handle single_config flag
    if args.single_config:
        dataset_config['configs'] = dataset_config['configs'][:1]
        print("Running single config for quick testing")
    
    # Base configuration
    base_cfg = {
        'epochs': args.epochs,
        'warmup': 10,
        'batch_size': 256,
        'lr': 0.03,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'tau': 3.0,
        'unlabeled_tau': 2.0,
        'use_ema': False,
        'ema_decay': 0.999,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_workers': 4,
        'seed': 42,
        'print_interval': 10
    }
    
    # Update with data specific settings
    base_cfg.update({
        'dataset': dataset_config['dataset'],
        'model_name': dataset_config['model_name'],
        'in_ch': dataset_config['in_ch']
    })
    
    # Create save dir
    save_dir = f"ablation_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Ablation configs Option A - (flags_tuple, name) pairs
    ablation_configs = [
        ((True, True, True), "SPMI"),           # Full model
        ((False, True, True), "SPMI w/o init"), # No initialization
        ((True, False, True), "SPMI w/o LG"),   # No label generation
        ((True, True, False), "SPMI w/o LC")    # No label condensation
    ]
    
    # Run for each dataset configuration different num_labeled, partial_rate
    all_results = {}
    for data_config in dataset_config['configs']:
        print(f"\n{'='*60}")
        print(f"Running ablation for {args.dataset}: l={data_config['num_labeled']}, p={data_config['partial_rate']}")
        print(f"{'='*60}")
        
        # Update config with specific parameters
        cfg = base_cfg.copy()
        cfg.update(data_config)
        
        config_results = {}
        for flags, name in ablation_configs:
            print(f"\n--- {name} ---")
            acc = run_ablation(cfg, flags[:3], save_dir)
            config_results[name] = acc
        
        # Store results
        config_key = f"l{data_config['num_labeled']}_p{data_config['partial_rate']}"
        all_results[config_key] = config_results
    
    # Save summary
    summary_file = os.path.join(save_dir, "ablation_summary.json")
    with open(summary_file, 'w') as f:
        json.dump({
            'dataset': args.dataset,
            'results': all_results,
            'ablation_configs': [name for _, name in ablation_configs]
        }, f, indent=2)
    
    # Print summary table
    print(f"\n{'='*80}")
    print(f"ABLATION STUDY SUMMARY - {args.dataset.upper()}")
    print(f"{'='*80}")
    
    for config_key, results in all_results.items():
        print(f"\n{config_key}:")
        print("-" * 40)
        for method, acc in results.items():
            print(f"  {method:<15}: {acc:6.2f}%")
    
    # Save CSV summary
    csv_file = os.path.join(save_dir, "ablation_summary.csv")
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Save header row with method names from ablation_configs
        header = ['Configuration'] + [name for _, name in ablation_configs]
        writer.writerow(header)
        
        # Data rows
        for config_key, results in all_results.items():
            row = [config_key] + [results.get(name, 0) for _, name in ablation_configs]
            writer.writerow(row)
    
    print(f"\nResults saved to {save_dir}")
    return all_results


def main():
    parser = argparse.ArgumentParser(description='SPMI Ablation Study')
    parser.add_argument('--dataset', 
                        choices=['fmnist', 'cifar10', 'cifar100', 'svhn'],
                        required=True,
                        help='Dataset to run ablation study on')
    parser.add_argument('--epochs', 
                        type=int, 
                        default=500,
                        help='Number of training epochs')
    parser.add_argument('--single_config',
                        action='store_true',
                        help='Run only first configuration (for quick testing)')
    
    args = parser.parse_args()
    
    print(f"Starting SPMI ablation study for {args.dataset}")
    print(f"Epochs: {args.epochs}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # If single_config, modify to run only one configuration
    if args.single_config:
        print("Single config flag detected")
    
    # Run ablation study
    results = run_full_ablation_study(args)
    
    print("\nAblation study completed successfully")


if __name__ == '__main__':
    main()