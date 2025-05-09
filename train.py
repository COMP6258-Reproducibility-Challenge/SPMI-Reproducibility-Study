import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from spmi import SPMI
from model import WideResNet, LeNet

def train(model, dataloader, optimizer, scheduler, device, num_epochs=500, warmup_epochs=10, tau=2.0, unlabeled_tau=2.0, use_ema=True, 
          ema_decay=0.999, eval_interval=10):
    # InitSPMI algorithm
    num_classes = model.fc2.out_features if hasattr(model, 'fc2') else model.fc.out_features
    spmi = SPMI(model, num_classes, tau, unlabeled_tau)
    
    # Create EMA model if needed
    ema_model = None
    if use_ema:
        ema_model = create_ema_model(model)
    
    # Save original candidate masks for labeled data
    dataset = dataloader.dataset
    original_masks = dataset.get_candidate_masks().to(device)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        
        # Initialize unlabeled data after warmup period
        if epoch == warmup_epochs:
            print(f"Epoch {epoch}: Initializing unlabeled data")
            spmi.initialize_unlabeled(dataloader, device)
        
        # Train for one epoch
        avg_loss = spmi.train_epoch(
            dataloader, optimizer, device, epoch, warmup_epochs, original_masks
        )
        
        # Update learning rate
        scheduler.step()
        
        # Update EMA model
        if use_ema:
            update_ema_model(model, ema_model, ema_decay)
        
        # Evaluate at regular intervals
        if epoch % eval_interval == 0 or epoch == num_epochs - 1:
            # Use EMA model for evaluation if available
            eval_model = ema_model if ema_model is not None else model
            accuracy = evaluate(eval_model, dataloader, device)
            print(f"Epoch {epoch}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2f}%")


def create_ema_model(model):
    # device err bruh
    device = next(model.parameters()).device
    
    if isinstance(model, WideResNet):
        # Create WideResNet with same parameters
        depth = model.depth if hasattr(model, 'depth') else 28
        widen_factor = model.widen_factor if hasattr(model, 'widen_factor') else 2
        dropout_rate = model.dropout_rate if hasattr(model, 'dropout_rate') else 0.0
        num_classes = model.fc.out_features if hasattr(model, 'fc') else 10
        
        ema_model = WideResNet(depth=depth, widen_factor=widen_factor, 
                              dropout_rate=dropout_rate, num_classes=num_classes)
    elif isinstance(model, LeNet):
        # Create LeNet
        num_classes = model.fc3.out_features if hasattr(model, 'fc3') else 10
        ema_model = LeNet(num_classes=num_classes)
    else:
        # Generic approach - might not work for all models
        ema_model = type(model)()
        
    # Move to the same device as the original model
    ema_model = ema_model.to(device)
    
    # Copy state dict
    ema_model.load_state_dict(model.state_dict())
    
    # Detach parameters
    for param in ema_model.parameters():
        param.detach_()
        
    return ema_model

def update_ema_model(model, ema_model, decay):
    with torch.no_grad():
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)


def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for imgs, _, targets, _, _ in dataloader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            
            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)
            
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    return 100 * correct / total