import torch
import torch.nn as nn
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

def setup_simple_experiment(batch_size=64):
    """
    Setup a small-scale experiment for debugging label behavior
    """
    print("Setting up label diagnostics experiment...")
    
    # Use CIFAR-10 with a small number of labeled examples
    dataset_name = 'cifar10'
    num_labeled = 100  # Small number for easier visualization
    
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

def visualize_candidate_labels(dataset, indices_to_track=None, save_dir='label_diagnostics'):
    """
    Visualize the candidate label sets for a few selected instances
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # If no indices provided, select a few from labeled and unlabeled sets
    if indices_to_track is None:
        # Get 5 labeled and 5 unlabeled indices
        labeled_indices = dataset.labeled_indices[:5]
        unlabeled_indices = dataset.unlabeled_indices[:5]
        indices_to_track = labeled_indices + unlabeled_indices
    
    # Get candidate labels for the selected indices
    candidate_sets = {}
    true_labels = {}
    for idx in indices_to_track:
        candidate_sets[idx] = dataset.candidate_labels[idx]
        true_labels[idx] = dataset.targets[idx]
    
    # Print information
    print("\nCurrent Candidate Label Sets:")
    print("-" * 50)
    for idx in indices_to_track:
        is_labeled = idx in dataset.labeled_indices
        type_str = "Labeled" if is_labeled else "Unlabeled"
        print(f"Instance {idx} ({type_str}):")
        print(f"  True Label: {true_labels[idx]}")
        print(f"  Candidate Labels: {candidate_sets[idx]}")
        print(f"  Num Candidates: {len(candidate_sets[idx])}")
        print()
    
    # Create visualization
    plt.figure(figsize=(12, len(indices_to_track) * 0.8))
    
    for i, idx in enumerate(indices_to_track):
        # Create a binary mask for the candidate set
        mask = np.zeros(dataset.num_classes)
        for label in candidate_sets[idx]:
            mask[label] = 1
        
        plt.subplot(len(indices_to_track), 1, i + 1)
        plt.bar(range(dataset.num_classes), mask, color='skyblue')
        plt.axvline(x=true_labels[idx], color='red', linestyle='--', label='True Label')
        plt.yticks([0, 1])
        plt.title(f"Instance {idx} ({'Labeled' if idx in dataset.labeled_indices else 'Unlabeled'})")
        
        if i == len(indices_to_track) - 1:
            plt.xlabel('Class Index')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/candidate_sets.png")

def track_label_evolution(dataset, indices_to_track, spmi, dataloader, device, 
                         num_epochs=3, warmup_epochs=1, save_dir='label_diagnostics'):
    """
    Track how candidate labels evolve during training for specific instances
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Store the original candidate sets
    original_sets = {}
    for idx in indices_to_track:
        original_sets[idx] = dataset.candidate_labels[idx].copy()
    
    # Store true labels
    true_labels = {}
    for idx in indices_to_track:
        true_labels[idx] = dataset.targets[idx]
    
    # Track candidate sets over epochs
    history = {idx: [original_sets[idx].copy()] for idx in indices_to_track}
    
    # Get batch information for tracked indices
    batch_info = {}
    for idx in indices_to_track:
        batch_info[idx] = []
    
    # Find which batch each tracked index is in
    for batch_idx, (_, _, _, indices, _) in enumerate(dataloader):
        for idx in indices:
            if idx.item() in indices_to_track:
                batch_info[idx.item()].append(batch_idx)
    
    # Print batch information
    print("\nBatch Information for Tracked Indices:")
    for idx, batches in batch_info.items():
        print(f"Instance {idx} appears in batches: {batches}")
    
    # Original candidate masks for PL data
    original_masks = dataset.get_candidate_masks().to(device)
    
    # Training loop
    model = spmi.model
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        
        # Initialize unlabeled data after warmup
        if epoch == warmup_epochs:
            print("Initializing unlabeled data...")
            spmi.initialize_unlabeled(dataloader, device)
            
            # Record candidate sets after initialization
            for idx in indices_to_track:
                if idx in dataset.unlabeled_indices:
                    history[idx].append(dataset.candidate_labels[idx].copy())
        
        # Train for one epoch
        spmi.train_epoch(dataloader, optimizer, device, epoch, warmup_epochs, original_masks)
        
        # Record candidate sets after this epoch
        for idx in indices_to_track:
            history[idx].append(dataset.candidate_labels[idx].copy())
        
        # Visualize candidate sets after this epoch
        visualize_candidate_labels(dataset, indices_to_track, 
                                  save_dir=f"{save_dir}/epoch_{epoch+1}")
    
    # Create evolution visualization
    for idx in indices_to_track:
        plt.figure(figsize=(12, 4))
        
        # Calculate the number of epochs to plot
        # Add 1 for initial state, add 1 if unlabeled and warmup for the post-initialization state
        num_epochs_to_plot = num_epochs + 1
        if idx in dataset.unlabeled_indices and warmup_epochs < num_epochs:
            num_epochs_to_plot += 1
        
        # Create a matrix of binary values for each class over epochs
        evolution_matrix = np.zeros((num_epochs_to_plot, dataset.num_classes))
        
        # Fill in the matrix
        for epoch_idx, candidates in enumerate(history[idx]):
            if epoch_idx < num_epochs_to_plot:  # Ensure we don't exceed matrix dimensions
                for label in candidates:
                    evolution_matrix[epoch_idx, label] = 1
        
        # Plot the matrix
        plt.imshow(evolution_matrix, cmap='Blues', aspect='auto')
        plt.colorbar(ticks=[0, 1], label='In Candidate Set')
        plt.xlabel('Class Index')
        plt.ylabel('Epoch')
        plt.yticks(range(num_epochs_to_plot))
        plt.title(f"Label Evolution for Instance {idx}")
        
        # Highlight true label
        true_label = true_labels[idx]
        plt.axvline(x=true_label, color='red', linestyle='--', label='True Label')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/evolution_instance_{idx}.png")
    
    # Create a summary of how often the true label is in the candidate set
    true_label_presence = {}
    for idx in indices_to_track:
        true_label = true_labels[idx]
        presence = []
        for epoch_candidates in history[idx]:
            presence.append(true_label in epoch_candidates)
        true_label_presence[idx] = presence
    
    print("\nTrue Label Presence in Candidate Sets:")
    print("-" * 50)
    for idx, presence in true_label_presence.items():
        is_labeled = idx in dataset.labeled_indices
        type_str = "Labeled" if is_labeled else "Unlabeled"
        print(f"Instance {idx} ({type_str}):")
        print(f"  True Label: {true_labels[idx]}")
        print(f"  Presence: {presence}")
        print(f"  Percentage: {sum(presence) / len(presence) * 100:.1f}%")
        print()

if __name__ == "__main__":
    # Setup
    dataset, dataloader, model, optimizer, spmi, device = setup_simple_experiment()
    
    # Select some indices to track
    labeled_indices = dataset.labeled_indices[:3]  # First 3 labeled instances
    unlabeled_indices = dataset.unlabeled_indices[:3]  # First 3 unlabeled instances
    indices_to_track = labeled_indices + unlabeled_indices
    
    # Visualize initial candidate sets
    visualize_candidate_labels(dataset, indices_to_track, save_dir='label_diagnostics/initial')
    
    # Track label evolution during training
    track_label_evolution(dataset, indices_to_track, spmi, dataloader, device, 
                         num_epochs=5, warmup_epochs=2, save_dir='label_diagnostics')