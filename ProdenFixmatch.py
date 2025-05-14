import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from dataset import SPMIDataset, get_transforms
from model import get_model

#Dataset wrapper that applies both weak and strong augmentations
class DualTransformDataset:
    def __init__(self, dataset, transform_weak, transform_strong):
        self.dataset = dataset
        self.transform_weak = transform_weak
        self.transform_strong = transform_strong
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Get original data (PIL) but temporarily disable transform
        original_transform = self.dataset.transform
        self.dataset.transform = None
        
        img, mask, target, idx, is_labeled = self.dataset.__getitem__(idx)
        
        # Restore original transform
        self.dataset.transform = original_transform
        
        # Apply both transforms
        img_weak = self.transform_weak(img)
        img_strong = self.transform_strong(img)
        
        return img_weak, img_strong, mask, target, idx, is_labeled
# Optimized PRODEN + FixMatch
class PRODEN_FixMatch_Fast:
    def __init__(self, model, candidate_labels, num_classes,
                 threshold=0.95, lam_ssl=1.0, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.num_classes = num_classes
        self.threshold = threshold
        self.lam_ssl = lam_ssl
        self.ce = nn.CrossEntropyLoss(reduction='none')
        
        # Prebuild candidate masks and soft labels
        self.build_candidate_structures(candidate_labels)
    # Prebuild vectorized structures 
    def build_candidate_structures(self, candidate_labels):
        
        n_samples = len(candidate_labels)
        
        # Create candidate masks n_samples, n_classes
        self.candidate_masks = torch.zeros(n_samples, self.num_classes, 
                                         dtype=torch.bool, device=self.device)
        # Initialize soft labels
        self.q = torch.zeros(n_samples, self.num_classes, device=self.device)
        
        for i, cands in enumerate(candidate_labels):
            if cands:
                self.candidate_masks[i, cands] = True
                self.q[i, cands] = 1.0 / len(cands)
    # Vectorized soft label update
    def update_soft_labels_vectorized(self, logits, indices):
        with torch.no_grad():
            probs = torch.softmax(logits, dim=1)
            
            # Extract relevant candidate masks
            batch_masks = self.candidate_masks[indices]  # (batch_size , n_classes)
            
            # Compute probabilities for candidates only
            candidate_probs = probs * batch_masks.float()
            
            # Renorm
            row_sums = candidate_probs.sum(dim=1, keepdim=True)
            row_sums = row_sums.clamp(min=1e-8)
            normalized_probs = candidate_probs / row_sums
            
            # Update soft labels for this batch
            self.q[indices] = normalized_probs
    
    # Vectorized PRODEN loss computation
    def compute_proden_loss_vectorized(self, logits, indices, is_labeled):

        # Only process labeled samples
        labeled_mask = is_labeled.bool()
        if not labeled_mask.any():
            return torch.tensor(0.0, device=self.device)
        
        labeled_indices = indices[labeled_mask]
        labeled_logits = logits[labeled_mask]
        
        # Get soft labels and candidate masks for labeled samples
        q_labeled = self.q[labeled_indices]
        candidate_masks = self.candidate_masks[labeled_indices]
        
        # Compute log probabilities
        log_probs = torch.log_softmax(labeled_logits, dim=1)
        
        # Apply masks to get only candidate probabilities
        masked_log_probs = log_probs * candidate_masks.float()
        masked_q = q_labeled * candidate_masks.float()
        
        # Compute weighted crossentropy
        losses = -(masked_q * masked_log_probs).sum(dim=1)
        
        return losses.mean()
    
    def train_epoch(self, loader, optimizer, scaler=None):
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in loader:
            imgs_weak, imgs_strong, masks, targets, indices, is_labeled = batch
            imgs_weak = imgs_weak.to(self.device, non_blocking=True)
            imgs_strong = imgs_strong.to(self.device, non_blocking=True)
            indices = indices.to(self.device, non_blocking=True) 
            is_labeled = is_labeled.to(self.device, non_blocking=True)  
            
            # Use mixed precision if scaler provided
            if scaler is not None:
                with autocast():
                    # Forward pass with both augmentations
                    logits_weak = self.model(imgs_weak)
                    logits_strong = self.model(imgs_strong)
                    
                    # Update soft labels using weak augmentation
                    self.update_soft_labels_vectorized(logits_weak, indices)
                    
                    # Compute PRODEN loss (only on labeled data)
                    sup_loss = self.compute_proden_loss_vectorized(logits_weak, indices, is_labeled)
                    
                    # Compute FixMatch SSL loss (on all data)
                    with torch.no_grad():
                        probs_weak = torch.softmax(logits_weak, dim=1)
                        max_probs, pseudo_labels = probs_weak.max(dim=1)
                        mask = max_probs.ge(self.threshold).float()
                    
                    ssl_loss = (self.ce(logits_strong, pseudo_labels) * mask).mean()
                    
                    # Total loss
                    loss = sup_loss + self.lam_ssl * ssl_loss
                
                # Optimization step with mixed precision
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                # Regular precision training
                logits_weak = self.model(imgs_weak)
                logits_strong = self.model(imgs_strong)
                
                # Update soft labels using weak augmentation
                self.update_soft_labels_vectorized(logits_weak, indices)
                
                # Compute PRODEN loss (only labeled)
                sup_loss = self.compute_proden_loss_vectorized(logits_weak, indices, is_labeled)
                
                # Compute FixMatch SSL loss (all data)
                with torch.no_grad():
                    probs_weak = torch.softmax(logits_weak, dim=1)
                    max_probs, pseudo_labels = probs_weak.max(dim=1)
                    mask = max_probs.ge(self.threshold).float()
                
                ssl_loss = (self.ce(logits_strong, pseudo_labels) * mask).mean()
                
                # Total loss
                loss = sup_loss + self.lam_ssl * ssl_loss
                
                # Optimization step
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        correct = 0
        total = 0
        
        for imgs, targets in loader:  # Test loader returns (imgs, targets)
            imgs = imgs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            outputs = self.model(imgs)
            _, predicted = outputs.max(1)
            
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        return 100.0 * correct / total

def main():
    parser = argparse.ArgumentParser(description="Optimized PRODEN + FixMatch baseline")
    parser.add_argument('--dataset', choices=['cifar10','cifar100','fashion_mnist','svhn'], default='cifar10')
    parser.add_argument('--root', default='./data')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_labeled', type=int, default=4000)
    parser.add_argument('--partial_rate', type=float, default=0.3)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.03)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--threshold', type=float, default=0.95)
    parser.add_argument('--lam_ssl', type=float, default=1.0)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--use_amp', action='store_true', help='Use automatic mixed precision')
    parser.add_argument('--compile', action='store_true', help='Use torch.compile (PyTorch 2.0+)')
    args = parser.parse_args()
    
    # Performance optimizations
    if torch.cuda.is_available():
        # Enable TensorFloat32 for faster training 
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        
        # Increase batch size for better GPU utilization
        args.batch_size = min(args.batch_size * 2, 1024)
        print(f"CUDA available. Increased batch size to: {args.batch_size}")
    
    # Create base dataset
    dataset = SPMIDataset(
        dataset_name=args.dataset,
        root=args.root,
        num_labeled=args.num_labeled,
        partial_rate=args.partial_rate,
        transform=None,  # No transform initially
        download=True,
        seed=42
    )
    
    # Create transforms
    transform_weak = get_transforms(args.dataset, strong_aug=False)
    transform_strong = get_transforms(args.dataset, strong_aug=True)
    
    # Wrap dataset with dual transforms
    train_dataset = DualTransformDataset(dataset, transform_weak, transform_strong)
    
    # Optimized DataLoader settings
    num_workers = min(16, os.cpu_count()) if torch.cuda.is_available() else 4
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,  # Keep workers alive between epochs
        prefetch_factor=4 if num_workers > 0 else 2,  # Load more batches ahead
        drop_last=True  # Ensures consistent batch sizes
    )
    
    # Test loader apply weak transform
    test_dataset = dataset.test_ds
    test_dataset.transform = transform_weak
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Get model
    in_ch = 1 if args.dataset == 'fashion_mnist' else 3
    if args.dataset in ['cifar10', 'svhn']:
        model_name = 'wrn-28-2'
    else:
        model_name = 'wrn-28-8'
    
    model = get_model(name=model_name, num_classes=dataset.num_classes, in_channels=in_ch)
    
    # Compile model for faster execution
    if args.compile and hasattr(torch, 'compile'):
        print("Compiling model...")
        model = torch.compile(model, mode="reduce-overhead")
    
    # Optimizer with proper weight decay handling
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True  # Often performs better
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)
    
    # Mixed precision scaler
    scaler = GradScaler() if args.use_amp and torch.cuda.is_available() else None
    
    # Training
    trainer = PRODEN_FixMatch_Fast(
        model,
        dataset.candidate_labels,
        dataset.num_classes,
        threshold=args.threshold,
        lam_ssl=args.lam_ssl,
        device=args.device
    )
    
    print("Starting optimized PRODEN + FixMatch training")
    print(f"Settings: {args}")
    print(f"Using mixed precision: {scaler is not None}")
    print(f"Number of workers: {num_workers}")
    
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        # Training
        loss = trainer.train_epoch(train_loader, optimizer, scaler)
        scheduler.step()
        
        # Evaluation
        if epoch % 10 == 0 or epoch == args.epochs:
            test_acc = trainer.evaluate(test_loader)
            
            if test_acc > best_acc:
                best_acc = test_acc
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'test_acc': test_acc,
                    'args': args
                }, f"best_proden_fixmatch_{args.dataset}.pth")
            
            print(f"Epoch {epoch:3d}/{args.epochs} | "
                  f"Loss: {loss:.4f} | "
                  f"Test Acc: {test_acc:.2f}% | "
                  f"Best: {best_acc:.2f}% | "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}")
    
    print(f"\nTraining completed! Best accuracy: {best_acc:.2f}%")

if __name__ == '__main__':
    main()