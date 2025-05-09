import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import argparse

# Import our implementations
from dataset import SPMIDataset, get_transforms
from model import LeNet, WideResNet
from spmi import SPMI

def test_dataset(dataset_name='cifar10', num_labeled=100, partial_rate=0.3, batch_size=32):
    #test if datset is loading correctly
    print(f"Testing dataset: {dataset_name}")
    
    transform = get_transforms(dataset_name, strong_aug=False)
    
    dataset = SPMIDataset(
        dataset_name=dataset_name,
        root='./data',
        num_labeled=num_labeled,
        partial_rate=partial_rate,
        transform=transform,
        download=True,
        seed=42
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    
    # Test a batch
    for imgs, candidate_masks, targets, indices, is_labeled in dataloader:
        print(f"Batch shape: {imgs.shape}")
        print(f"Candidate masks shape: {candidate_masks.shape}")
        print(f"Targets shape: {targets.shape}")
        print(f"Indices shape: {indices.shape}")
        print(f"Is labeled shape: {is_labeled.shape}")
        
        print(f"Number of labeled instances: {is_labeled.sum().item()}")
        print(f"Number of unlabeled instances: {len(is_labeled) - is_labeled.sum().item()}")
        
        labeled_masks = candidate_masks[is_labeled == 1]
        if len(labeled_masks) > 0:
            print(f"Average number of candidates per labeled instance: {labeled_masks.sum(dim=1).float().mean().item():.2f}")
        
        break
    
    print("Dataset test passed!\n")
    return dataset, dataloader

def test_model(dataset_name='cifar10', num_classes=10):
    """Test if the model initializes correctly"""
    print(f"Testing model for {dataset_name}")
    # Create appropriate model
    if dataset_name == 'fashion_mnist':
        model = LeNet(num_classes=num_classes)
    elif dataset_name == 'cifar100':
        model = WideResNet(depth=28, widen_factor=8, num_classes=num_classes)
    else:  # cifar10 or svhn
        model = WideResNet(depth=28, widen_factor=2, num_classes=num_classes)
    # Print model summary
    print(f"Model type: {type(model).__name__}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    # Test forward pass with dummy input
    if dataset_name == 'fashion_mnist':
        dummy_input = torch.randn(1, 1, 28, 28)
    else:
        dummy_input = torch.randn(1, 3, 32, 32)
    # Try forward pass
    try:
        output = model(dummy_input)
        print(f"Output shape: {output.shape}")
        print("Model test passed!\n")
        return model
    except Exception as e:
        print(f"Error in forward pass: {e}")
        return None
def test_spmi_training(model, dataloader, device, num_epochs=2):
    """Test a few training epochs with SPMI"""
    print("Testing SPMI training")
    
    # Move model to device
    model = model.to(device)
    
    # Create optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.9,
        weight_decay=5e-4
    )
    
    # Create scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs
    )
    
    # Initialize SPMI
    num_classes = model.fc.out_features if hasattr(model, 'fc') else model.fc3.out_features
    spmi = SPMI(model, num_classes=num_classes)
    
    # Train for a few epochs
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Initialize unlabeled data if it's the first epoch
        if epoch == 0:
            print("Initializing unlabeled data")
            spmi.initialize_unlabeled(dataloader, device)
        
        # Train for one epoch
        model.train()
        total_loss = 0.0
        
        for batch_idx, (imgs, candidate_masks, _, indices, is_labeled) in enumerate(dataloader):
            imgs = imgs.to(device)
            candidate_masks = candidate_masks.to(device)
            is_labeled = is_labeled.to(device)
            
            # Forward pass
            outputs = model(imgs)
            
            # Calculate loss
            loss = spmi.calculate_loss(outputs, candidate_masks)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update class priors
            spmi.update_class_priors(outputs.detach())
            
            # Update candidate labels (simplified for test)
            with torch.no_grad():
                # Expand labels
                updated_masks = spmi.expand_labels(outputs, candidate_masks, is_labeled)
                
                # Condense labels
                updated_masks = spmi.condense_labels(outputs, updated_masks, is_labeled)
                
                # Update dataset's candidate labels
                for i, idx in enumerate(indices):
                    dataloader.dataset.update_candidate_labels(idx.item(), updated_masks[i])
            
            total_loss += loss.item()
            
            # Print progress
            if (batch_idx + 1) % 10 == 0:
                print(f"Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}")
        
        # Update scheduler
        scheduler.step()
        
        # Calculate average loss
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
    
    print("SPMI training test passed!\n")
    return True

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test dataset
    dataset, dataloader = test_dataset(dataset_name='cifar10', num_labeled=100)
    
    # Test model
    model = test_model(dataset_name='cifar10', num_classes=10)
    
    if model is not None:
        # Test SPMI training
        test_spmi_training(model, dataloader, device, num_epochs=2)

if __name__ == "__main__":
    main()