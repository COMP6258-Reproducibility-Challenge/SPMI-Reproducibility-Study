import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
try:
    import cupy as cp
    USE_CUPY = True
    print("Using CuPy for GPU acceleration")
except ImportError:
    USE_CUPY = False
    print("CuPy not found, using NumPy instead")

class SPMI:
    #init loss eq(1-2), unlabeled data eq(3), label expansion eq(11), label condensation eq(15) check paper for equations
    # The MI inplementation needs a little bit of work i think
    def __init__(self, model, num_classes, tau=2.0, unlabeled_tau=2.0):
        self.model = model
        self.num_classes = num_classes
        self.tau = tau
        self.unlabeled_tau = unlabeled_tau
        # init uniform prior
        device = next(model.parameters()).device
        self.class_priors = (torch.ones(num_classes) / num_classes).to(device)
    def initialize_unlabeled(self, dataloader, device):
            #eq(3) we pseudo candidate label sets for unlabeled data
            # for xi, if fk(xi) > 1/c, then the loss will be smaller than the average and otherwise, the losser is larger if so the label is not the true label
            self.model.eval()
            dataset = dataloader.dataset
            with torch.no_grad():
                for imgs, candidate_masks, _, indices, is_labeled in dataloader:
                    # Only process unlabeled data
                    unlabeled_mask = (is_labeled == 0)
                    if not torch.any(unlabeled_mask):
                        continue
                    # Get model predictions
                    imgs = imgs[unlabeled_mask].to(device)
                    unlabeled_indices = indices[unlabeled_mask]
                    outputs = self.model(imgs)
                    probs = F.softmax(outputs, dim=1)
                    # find labels with probability > 1/c for each inst
                    threshold = 1.0 / self.num_classes
                    for i, idx in enumerate(unlabeled_indices):
                        # Find candidate labels with prob > 1/c
                        candidate_labels = (probs[i] > threshold).nonzero(as_tuple=True)[0].tolist()
                        # If no labels meet the criteria, include the top label
                        if len(candidate_labels) == 0:
                            top_label = probs[i].argmax().item()
                            candidate_labels = [top_label]
                        # Make sure we have at least one candidate but not all classes
                        if len(candidate_labels) == self.num_classes:
                            # If all classes are candidates thenn keep only the top half
                            top_values, top_indices = torch.topk(probs[i], k=self.num_classes//2)
                            candidate_labels = top_indices.tolist()
                        # Update the candidate labels
                        dataset.update_candidate_labels(idx.item(), candidate_labels)
    
    def update_class_priors(self, outputs):
        # eq(16) update class priors, acc paper, we approx priors using : uj = (1/n) sigma_{i=1}^n fj(xi) ::: basiclly the mean of the model outputs across all instances
        probs = F.softmax(outputs, dim=1)
        new_priors = probs.mean(dim=0)
        # DEVICE ERRRRRRROROROROROROORR
        if self.class_priors.device != new_priors.device:
            new_priors = new_priors.to(self.class_priors.device)        
        # update class priors with exponential moving average for stability (out of paper scope but works well)
        self.class_priors = 0.9 * self.class_priors + 0.1 * new_priors
        return self.class_priors

    
    def expand_labels(self, outputs, candidate_masks, is_labeled, original_masks=None):
            #label expansion eq(11)... paper says a label k should be added to candidate set S if log(q(k|zi)/p(k)) > 0 iff q(k|zi) > p(k)
            # q(k|zi) is the model's output probability for class k, and p(k) is the prior probability for class k
            probs = F.softmax(outputs, dim=1)
            updated_masks = candidate_masks.clone()
            for i in range(len(probs)):
                for k in range(self.num_classes):
                    # if label is alr in the candidate set, skip
                    if candidate_masks[i, k] == 1:
                        continue
                    # add label if q(k|z_i) > p(k)
                    if probs[i, k] > self.class_priors[k]:
                        # For labeled data only add labels from original candidate set
                        if is_labeled[i] == 1:
                            if original_masks is not None and original_masks[i, k] == 1:
                                updated_masks[i, k] = 1
                        else:
                            # For unlabeled data add any label meeting the criterion
                            updated_masks[i, k] = 1
            
            return updated_masks
    
    def condense_labels(self, outputs, candidate_masks, is_labeled):
            # Removes unlikely labels based on information score function 
            # the paper gives the info function G as : G(xi, Si, k) = D_KL[(f_{Si\\k}(xi)||f_{Si}(xi)
            probs = F.softmax(outputs, dim=1)
            updated_masks = candidate_masks.clone()
            # for each inst
            for i in range(len(probs)):
                # get the candidate labels
                candidates = candidate_masks[i].nonzero(as_tuple=True)[0]
                #if there's only one or no candidate labels skip
                if len(candidates) <= 1:
                    continue
                # set the init MI scores
                min_score = float('inf')
                min_label = None
                max_score = -float('inf')
                for k in candidates:
                    # Create mask without label k
                    reduced_mask = candidate_masks[i].clone()
                    reduced_mask[k] = 0
                    # Get remaining candidates
                    remaining = reduced_mask.nonzero(as_tuple=True)[0]
                    # Skip if removing k would leave no candidates
                    if len(remaining) == 0:
                        continue
                    # Get probabilities for Si and Si without k
                    p_si = probs[i, candidates]
                    p_si_without_k = probs[i, remaining]
                    # normalize probabilities
                    p_si = p_si / (p_si.sum() + 1e-10)
                    p_si_without_k = p_si_without_k / (p_si_without_k.sum() + 1e-10)
                    # get indices of remaining candidates in original candidates list
                    remaining_indices = []
                    for idx, c in enumerate(candidates):
                        if c != k:
                            remaining_indices.append(idx)
                    # Calculate KL 
                    kl_div = F.kl_div(
                        torch.log(p_si_without_k + 1e-10),
                        p_si[remaining_indices],
                        reduction='sum'
                    )
                    score = kl_div.item()
                    # Update min and max scores
                    if score < min_score:
                        min_score = score
                        min_label = k.item()
                    if score > max_score:
                        max_score = score
                # Apply threshold for removing label
                threshold = self.tau if is_labeled[i] == 1 else self.unlabeled_tau
                if max_score > threshold and min_label is not None:
                    # Only remove if there will still be candidates left
                    if updated_masks[i].sum() > 1:
                        updated_masks[i, min_label] = 0
            return updated_masks
    
    def calculate_loss(self, outputs, candidate_masks):
            #loss eq(1-2) in the paper
            # check paper too long to type
            # Get probabilities from model outputs with proper numerical stability
            log_probs = F.log_softmax(outputs, dim=1)
            probs = F.softmax(outputs, dim=1)
            batch_size = outputs.size(0)
            # Initialize loss directly with the outputs to maintain gradient flow
            loss = outputs.new_zeros(1, requires_grad=True)
            # Clip outputs to prevent extreme values
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                print("NaN or Inf detected BEFORE LOSS CALC")
                outputs = torch.nan_to_num(outputs, nan=0.0, posinf=1e6, neginf=-1e6)
            valid_instances = 0
            instance_losses = []
            for i in range(batch_size):
                # Get candidate labels
                candidates = candidate_masks[i].nonzero(as_tuple=True)[0]
                # Skip instances with no candidates
                if len(candidates) == 0:
                    continue
                valid_instances += 1
                # Calculate weights according to(2) with numerical stability
                candidate_probs = probs[i, candidates]
                weights_sum = candidate_probs.sum() + 1e-10  # Avoid division by zero
                # Calculate weights
                weights = candidate_probs / weights_sum
                # Check for NaNs in weights
                if torch.isnan(weights).any():
                    print(f"NaN weights for instance {i}")
                    continue
                # Calculate negative log likelihood for each candidate with clipping
                candidate_nll = -torch.clamp(log_probs[i, candidates], min=-100.0, max=100.0)
                # Apply weights and sum
                instance_loss = torch.sum(weights * candidate_nll)
                # Check for NaN/Inf in instance loss
                if torch.isnan(instance_loss) or torch.isinf(instance_loss):
                    print(f"NaN/Inf instance loss for instance {i}")
                    print(f"Weights: {weights}")
                    print(f"Candidate NLL: {candidate_nll}")
                    continue
                instance_losses.append(instance_loss.item())
                # Add to total loss
                loss = loss + instance_loss
            # Ensure we don't divide by zero if all instances have no candidates
            if valid_instances > 0:
                loss = loss / valid_instances
            else:
                print("No valid insts in batch for loss calc")
            # Scale loss for better numerical stability, but not too much to cause overflow
            scale_factor = 1.0
            # Log loss statistics
            if len(instance_losses) > 0:
                print(f"Instance losses: min={min(instance_losses):.4f}, max={max(instance_losses):.4f}, mean={sum(instance_losses)/len(instance_losses):.4f}")
            
            return loss * scale_factor
    
    def train_epoch(self, dataloader, optimizer, device, epoch, warmup_epochs, original_candidate_masks=None):
        #ALgo 1 from the paper
        # 1. Train on labeled data during warmup
        # 2. Initialize unlabeled data after warmup
        # 3. Update candidate labels via expansion and condensation
        # 4. Update class priors
        # 5. Return average loss for the epoch
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        valid_losses = []
        for batch_idx, (imgs, candidate_masks, targets, indices, is_labeled) in enumerate(dataloader):
            # check for NANANANANNANANA
            if torch.isnan(imgs).any():
                print(f"Nan value here (train inital): {batch_idx}")
                continue
            imgs = imgs.to(device)
            candidate_masks = candidate_masks.to(device)
            is_labeled = is_labeled.to(device)
            # warmup skip batches with all unlabeled data
            if epoch < warmup_epochs and not torch.any(is_labeled == 1):
                continue
            # Forward
            try:
                outputs = self.model(imgs)
                # nan check
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    print(f"Nan here after forward{batch_idx}")
                    # skip
                    continue
                # calc loss
                # only labeled data for warmup
                if epoch < warmup_epochs:
                    # mask to keep labeled 
                    labeled_mask = (is_labeled == 1)
                    if labeled_mask.sum() > 0:
                        labeled_outputs = outputs[labeled_mask]
                        labeled_candidate_masks = candidate_masks[labeled_mask]
                        loss = self.calculate_loss(labeled_outputs, labeled_candidate_masks)
                    else:
                        continue  # skip batch if no labeled data
                else:
                    loss = self.calculate_loss(outputs, candidate_masks)
                # Skip batch if loss is NaN
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Nan {batch_idx}, skipping (THis is afetr loss calc)")
                    continue
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                optimizer.step()
                # debugging
                if batch_idx % 10 == 0:
                    print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
                # eq(16) update class priors
                self.update_class_priors(outputs.detach())
                # cadidate label update after warmup
                if epoch >= warmup_epochs:
                    with torch.no_grad():
                        try:
                            # MI label expansion - infinite pain
                            updated_masks = self.expand_labels(
                                outputs, candidate_masks, is_labeled, original_candidate_masks
                            )
                            # Condense labels using information 
                            updated_masks = self.condense_labels(
                                outputs, updated_masks, is_labeled
                            )
                            # Update candidate labels
                            for i, idx in enumerate(indices):
                                dataloader.dataset.update_candidate_labels(idx.item(), updated_masks[i])
                        except Exception as e:
                            print(f"ERROR during label update: {e}")
                # Track valid losses
                if not torch.isnan(loss) and not torch.isinf(loss):
                    loss_val = loss.item()
                    total_loss += loss_val
                    valid_losses.append(loss_val)
                    num_batches += 1
            except Exception as e:
                print(f"ERROR processing batch {batch_idx}: {e}")
        # Avoid division by zero
        if num_batches > 0:
            avg_loss = total_loss / num_batches
        else:
            avg_loss = float('nan')
            
        if valid_losses:
            print(f"Valid losses: min={min(valid_losses):.4f}, max={max(valid_losses):.4f}, mean={sum(valid_losses)/len(valid_losses):.4f}")
        else:
            print("No valid losses recorded for this epoch")
        return avg_loss