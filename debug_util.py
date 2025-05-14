import torch
import torch.nn.functional as F
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Analyse model outputs to help diagnose issues with the training
def analyze_model_outputs(model, dataloader, device, save_dir='debug_outputs', is_test_loader=False):

    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    all_outputs = []
    all_probs = []
    all_is_labeled = []
    all_candidate_counts = []
    
    # Collect model outputs
    with torch.no_grad():
        for batch in dataloader:
            if is_test_loader:
                # Test loader likely only has images and targets
                if len(batch) == 2:
                    imgs, _ = batch
                    candidate_masks = None
                    is_labeled = None
                else:
                    print(f"Unexpected batch format for test loader: {len(batch)} items")
                    continue
            else:
                # Training loader has all 5 items
                imgs, candidate_masks, _, _, is_labeled = batch
            
            imgs = imgs.to(device)
            try:
                outputs = model(imgs)
                
                # Check for NaN values
                if torch.isnan(outputs).any():
                    print("WARNING: NaN values detected in model outputs ")
                    # Replace NaN with zeros for analysis
                    outputs = torch.nan_to_num(outputs, nan=0.0)
                
                probs = F.softmax(outputs, dim=1)
                
                all_outputs.append(outputs.cpu())
                all_probs.append(probs.cpu())
                
                if is_labeled is not None:
                    all_is_labeled.append(is_labeled)
                else:
                    # For test data we assume all are labeled
                    all_is_labeled.append(torch.ones(len(imgs)))
                    
                if candidate_masks is not None:
                    all_candidate_counts.append(candidate_masks.sum(dim=1))
                else:
                    # For test data no candidate masks
                    all_candidate_counts.append(torch.ones(len(imgs)))
                    
            except Exception as e:
                print(f"Error processing batch: {e}")
    
    if len(all_outputs) == 0:
        print("No valid outputs to analyse")
        return {}