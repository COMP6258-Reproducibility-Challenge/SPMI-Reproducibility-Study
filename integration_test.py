import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import numpy as np

from dataset import SPMIDataset, get_transforms
from model import LeNet, WideResNet
from train import train
from spmi import SPMI
from debug_util import analyze_model_outputs

# Run a mini version of the full training pipeline to verify all components work together
def main():

    print("Starting SPMI Integration Test")
    
    # Configuration small scale for testing
    dataset_name = 'cifar10'
    num_labeled = 1000  # Use more labeled examples to improve initial training
    batch_size = 64
    num_epochs = 10  # Run more epochs to see progress
    warmup_epochs = 2
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs('debug_outputs', exist_ok=True)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Load dataset
    print(f"Loading {dataset_name} dataset with {num_labeled} labeled examples")
    transform = get_transforms(dataset_name, strong_aug=True)  # Use strong augmentation
    dataset = SPMIDataset(
        dataset_name=dataset_name,
        root='./data',
        num_labeled=num_labeled,
        partial_rate=0.3,
        transform=transform,
        download=True,
        seed=42
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True  # Drop last batch to avoid size issues
    )
    
    # Get test loader
    test_loader = dataset.get_test_loader(batch_size=batch_size)
    
    # Create model
    print("Creating model")
    model = WideResNet(depth=16, widen_factor=2, num_classes=10)
    model = model.to(device)
    
    # Analyze initial model outputs
    print("Analyzing initial model outputs...")
    analyze_model_outputs(model, dataloader, device, save_dir='debug_outputs/initial')
    
    # Create optimizer with better learning rate
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.03,  # Higher learning rate
        momentum=0.9,
        weight_decay=5e-4,  # Stronger regularization
        nesterov=True
    )
    
    # Create scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs
    )
    
    # Create SPMI instance
    spmi = SPMI(model, num_classes=10, tau=2.0, unlabeled_tau=1.5)
    
    # Training loop
    print(f"Starting training for {num_epochs} epochs with {warmup_epochs} warmup epochs")
    best_acc = 0.0
    
    # Extract original candidate masks for PL data
    original_masks = dataset.get_candidate_masks().to(device)
    
    for epoch in range(num_epochs):
        print(f"\n Epoch {epoch+1}/{num_epochs} ")
        
        # Initialize unlabeled data after warmup
        if epoch == warmup_epochs:
            print(f"Epoch {epoch+1}: Initializing unlabeled data")
            spmi.initialize_unlabeled(dataloader, device)
            
            # Analyse model outputs after initialization
            print("Analyzing model outputs after initialization...")
            analyze_model_outputs(model, dataloader, device, save_dir=f'debug_outputs/epoch_{epoch+1}_after_init')
        
        # Train for one epoch
        avg_loss = spmi.train_epoch(
            dataloader, optimizer, device, epoch, warmup_epochs, original_masks
        )
        
        # Update scheduler
        scheduler.step()
        
        # Evaluate on test set
        test_accuracy = evaluate(model, test_loader, device)
        
        # Print epoch summary
        print(f"Epoch {epoch+1}: Average Loss = {avg_loss:.4f}, Test Accuracy = {test_accuracy:.2f}%")
        
        # Save best model
        if test_accuracy > best_acc:
            best_acc = test_accuracy
            print(f"New best accuracy: {best_acc:.2f}%")
            
        # Analyse model outputs periodically
        if (epoch + 1) % 2 == 0 or epoch == num_epochs - 1:
            print(f"Analyzing model outputs after epoch {epoch+1}...")
            analyze_model_outputs(model, dataloader, device, save_dir=f'debug_outputs/epoch_{epoch+1}')
    
    print(f"Integration test completed! Best test accuracy: {best_acc:.2f}%")
    print("Performing final model analysis")
    analyze_model_outputs(model, dataloader, device, save_dir='debug_outputs/final')
    analyze_model_outputs(model, test_loader, device, save_dir='debug_outputs/final_test', is_test_loader=True)


def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    return 100.0 * correct / total


if __name__ == "__main__":
    main()