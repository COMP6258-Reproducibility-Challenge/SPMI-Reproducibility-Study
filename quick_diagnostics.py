import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os

# Set matplotlib backend to a non-GUI backend to avoid Qt issues
import matplotlib
matplotlib.use('Agg')  # Use Agg backend (non-interactive)
import matplotlib.pyplot as plt

from dataset import SPMIDataset, get_transforms
from model import WideResNet
from spmi import SPMI

def setup_simple_experiment(batch_size=16):
    """
    Setup a very small-scale experiment for quick diagnostics
    """
    print("Setting up quick diagnostics experiment...")
    
    # Use CIFAR-10 with a tiny number of labeled examples
    dataset_name = 'cifar10'
    num_labeled = 20  # Very small number for quick runs
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load dataset
    transform = get_transforms(dataset_name, strong_aug=False)  # Simple transforms
    dataset = SPMIDataset(
        dataset_name=dataset_name,
        root='./data',
        num_labeled=num_labeled,
        partial_rate=0.3,
        transform=transform,
        download=True,
        seed=42
    )
    
    # Create dataloader with small batch size
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # No shuffling for consistent indices
        num_workers=2,
        drop_last=False
    )
    
    # Create a simple model
    model = WideResNet(depth=16, widen_factor=2, num_classes=10)
    model = model.to(device)
    
    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # SPMI instance
    spmi = SPMI(model, num_classes=10)
    
    return dataset, dataloader, model, optimizer, spmi, device

def quick_label_test(num_epochs=2, warmup_epochs=1):
    """
    Run a quick test of label handling in SPMI
    """
    # Setup
    dataset, dataloader, model, optimizer, spmi, device = setup_simple_experiment()
    
    # Select a few indices to track (first 2 labeled and first 2 unlabeled)
    labeled_indices = dataset.labeled_indices[:2]
    unlabeled_indices = dataset.unlabeled_indices[:2]
    indices_to_track = labeled_indices + unlabeled_indices
    
    # Print initial candidate sets
    print("\nInitial Candidate Label Sets:")
    print("-" * 40)
    for idx in indices_to_track:
        label_type = "Labeled" if idx in dataset.labeled_indices else "Unlabeled"
        true_label = dataset.targets[idx]
        candidates = dataset.candidate_labels[idx]
        print(f"Instance {idx} ({label_type}):")
        print(f"  True Label: {true_label}")
        print(f"  Candidates: {candidates}")
    
    # Original candidate masks for labeled data
    original_masks = dataset.get_candidate_masks().to(device)
    
    # Short training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        
        # Initialize unlabeled data after warmup
        if epoch == warmup_epochs:
            print("Initializing unlabeled data...")
            spmi.initialize_unlabeled(dataloader, device)
            
            # Print candidate sets after initialization
            print("\nCandidate Sets After Initialization:")
            print("-" * 40)
            for idx in unlabeled_indices:
                true_label = dataset.targets[idx]
                candidates = dataset.candidate_labels[idx]
                print(f"Instance {idx} (Unlabeled):")
                print(f"  True Label: {true_label}")
                print(f"  Candidates: {candidates}")
                print(f"  True label in candidates: {true_label in candidates}")
        
        # Train for one epoch with minimal output
        print(f"Training epoch {epoch+1}...")
        total_loss = 0.0
        num_batches = 0
        
        for imgs, candidate_masks, _, indices, is_labeled in dataloader:
            imgs = imgs.to(device)
            candidate_masks = candidate_masks.to(device)
            is_labeled = is_labeled.to(device)
            
            # Forward pass
            outputs = model(imgs)
            
            # Calculate loss
            loss = spmi.calculate_loss(outputs, candidate_masks)
            
            # Update model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
                
            total_loss += loss.item()
            num_batches += 1
        
        # Print average loss
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            print(f"Average loss: {avg_loss:.4f}")
    
    # Print final candidate sets
    print("\nFinal Candidate Label Sets:")
    print("-" * 40)
    for idx in indices_to_track:
        label_type = "Labeled" if idx in dataset.labeled_indices else "Unlabeled"
        true_label = dataset.targets[idx]
        candidates = dataset.candidate_labels[idx]
        print(f"Instance {idx} ({label_type}):")
        print(f"  True Label: {true_label}")
        print(f"  Candidates: {candidates}")
        print(f"  True label in candidates: {true_label in candidates}")

if __name__ == "__main__":
    quick_label_test(num_epochs=2, warmup_epochs=1)