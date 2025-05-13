import torch
import torch.nn.functional as F

class SPMI:
    def __init__(self,
                 model,
                 num_classes,
                 tau: float = 3.0,
                 unlabeled_tau: float = 2.0,
                 init_threshold: float = None,
                 prior_alpha: float = 0.9,
                 use_ib_penalty: bool = True,
                 ib_beta: float = 0.01):
        """
        model:          your nn.Module
        num_classes:    C
        tau:            threshold for partial‐label data (condensation)
        unlabeled_tau:  threshold for unlabeled data (condensation)
        init_threshold: threshold for initialization (>f_k > init_threshold);
                        if None, defaults to 1/C
        prior_alpha:    EMA smoothing factor for class-priors (0 < alpha < 1)
        use_ib_penalty: whether to add β·H(f(x)) to the loss
        ib_beta:        coefficient for IB penalty
        """
        self.model = model
        self.num_classes = num_classes
        self.tau = tau
        self.unlabeled_tau = unlabeled_tau
        self.init_threshold = init_threshold or (1.0 / num_classes)
        self.prior_alpha = prior_alpha
        self.use_ib_penalty = use_ib_penalty
        self.ib_beta = ib_beta

        # initialize uniform priors μ_j = 1/C
        device = next(model.parameters()).device
        self.class_priors = torch.full((num_classes,), 1.0/num_classes, device=device)

    def initialize_unlabeled(self, dataloader, device):
        """
        Eq. 3 (with raised threshold): For each *unlabeled* x_i,
        add k iff softmax[f_k(x_i)] > init_threshold.
        """
        self.model.eval()
        dataset = dataloader.dataset
        with torch.no_grad():
            for imgs, _, _, indices, is_labeled in dataloader:
                imgs = imgs.to(device)
                probs = F.softmax(self.model(imgs), dim=1)
                for i, sample_idx in enumerate(indices):
                    if is_labeled[i].item() == 1:
                        continue
                    mask = (probs[i] > self.init_threshold).long()
                    candidates = mask.nonzero(as_tuple=True)[0].tolist()
                    # Fallback unlikely if init_threshold <= max(probs)
                    if not candidates:
                        topk = torch.topk(probs[i], k=2).indices.tolist()
                        candidates = topk
                    dataset.update_candidate_labels(sample_idx.item(), candidates)

    def expand_labels(self, outputs, candidate_masks, is_labeled, indices, original_masks=None):
        """
        Eq. 11: Add k iff q(k|z_i) > μ_k,
        but for labeled only if in original_masks.
        This version avoids CUDA boolean ambiguities by vectorizing the mask.
        """
        probs = F.softmax(outputs, dim=1)          # [B, C]
        updated = candidate_masks.clone()          # [B, C]

        for i, sample_idx in enumerate(indices):
            # 1) Get the row of probabilities and existing mask
            row_probs = probs[i]                   # shape [C]
            row_taken = updated[i].bool()          # shape [C]

            # 2) Build a mask of labels to skip:
            #    skip if already taken OR prob <= prior
            skip_mask = row_taken | (row_probs <= self.class_priors)

            # 3) The candidates to add are where skip_mask is False
            add_ks = torch.nonzero(~skip_mask, as_tuple=False).view(-1)

            # 4) For each such k, apply the labeled/unlabeled logic
            for k in add_ks.tolist():
                if is_labeled[i].item() == 1:
                    # only expand within the original labeled mask
                    if original_masks is None or original_masks[sample_idx.item(), k].item():
                        updated[i, k] = 1
                else:
                    # for unlabeled, always add
                    updated[i, k] = 1

        return updated



    def condense_labels(self, outputs, candidate_masks, is_labeled):
        """
        Eq.15: remove the label with smallest G if the largest G > tau.
        """
        probs = F.softmax(outputs, dim=1)
        updated = candidate_masks.clone()
        
        for i in range(outputs.size(0)):
            cands = candidate_masks[i].nonzero(as_tuple=True)[0]
            if cands.numel() <= 1:
                continue
                
            # 1) Full-set distribution Q (normalized once)
            p_full = probs[i, cands]
            p_full = p_full / (p_full.sum() + 1e-10)
            
            G_vals = []
            for j, k in enumerate(cands):
                # 2) Remaining distribution P (drop k, renormalize)
                idx = torch.cat([cands[:j], cands[j+1:]])
                p_remaining = probs[i, idx]
                p_remaining = p_remaining / (p_remaining.sum() + 1e-10)
                
                # 3) Original distribution Q without k (NO renormalization!)
                p_orig = torch.cat([p_full[:j], p_full[j+1:]])
                
                # 4) KL(P || Q) where Q is the original full-set probabilities
                G = F.kl_div(
                    torch.log(p_orig + 1e-10),
                    p_remaining,
                    reduction='sum'
                ).item()
                G_vals.append((G, k.item()))
                
            maxG, _ = max(G_vals, key=lambda x: x[0])  
            minG, min_k = min(G_vals, key=lambda x: x[0])
            thresh = self.tau if is_labeled[i].item() == 1 else self.unlabeled_tau 
            
            if maxG > thresh and updated[i].sum() > 1:
                updated[i, min_k] = 0
                
        return updated

    def calculate_loss(self, outputs, candidate_masks):
        """
        Eq. 1–2: Weighted negative log‐likelihood over candidates.
        Optional IB penalty if use_ib_penalty=True.
        """
        logp = F.log_softmax(outputs, dim=1)
        p = logp.exp()
        losses = []

        for i in range(outputs.size(0)):
            c = candidate_masks[i].nonzero(as_tuple=True)[0]
            if c.numel() == 0:
                continue
            w = p[i, c] / (p[i, c].sum() + 1e-10)
            nll = -logp[i, c]
            losses.append((w * nll).sum())

        if not losses:
            return torch.tensor(0.0, device=outputs.device, requires_grad=True)

        loss = torch.stack(losses).mean()
        if self.use_ib_penalty:
            H = -(p * logp).sum(dim=1).mean()
            loss = loss + self.ib_beta * H

        return loss

    def update_class_priors(self, dataloader, device):
        """
        Smoothed Eq. 16: EMA update of μ over all samples every epoch.
        """
        self.model.eval()
        total = torch.zeros(self.num_classes, device=device)
        count = 0

        with torch.no_grad():
            for imgs, *_ in dataloader:
                imgs = imgs.to(device)
                p = F.softmax(self.model(imgs), dim=1)
                total += p.sum(dim=0)
                count += p.size(0)

        new_priors = total / count
        # EMA smoothing
        self.class_priors = self.prior_alpha * self.class_priors + (1 - self.prior_alpha) * new_priors
        return self.class_priors

    def train_epoch(self,
                    dataloader,
                    optimizer,
                    device,
                    epoch: int,
                    warmup_epochs: int,
                    original_masks=None):
        """
        1) Warm-up on labeled only
        2) At epoch == warmup_epochs: initialize_unlabeled
        3) Each batch: compute loss, backward, step
        4) After batch (if past warm-up): expand+condense+update dataset
        5) At end: update_class_priors with EMA smoothing
        """
        self.model.train()
        warmup = (epoch < warmup_epochs)

        if epoch == warmup_epochs:
            self.initialize_unlabeled(dataloader, device)

        running_loss = 0.0
        n_batches = 0

        for imgs, cm, targets, indices, is_labeled in dataloader:
            imgs = imgs.to(device)
            cm = cm.to(device)
            is_labeled = is_labeled.to(device)

            if warmup and not is_labeled.any():
                continue

            outputs = self.model(imgs)
            if warmup:
                mask = cm[is_labeled == 1]
                out = outputs[is_labeled == 1]
            else:
                mask, out = cm, outputs

            loss = self.calculate_loss(out, mask)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
            optimizer.step()

            if not warmup:
                with torch.no_grad():
                    new_cm = self.expand_labels(outputs, cm, is_labeled, indices, original_masks)
                    new_cm = self.condense_labels(outputs, new_cm, is_labeled)
                    for i, sample_idx in enumerate(indices):
                        dataloader.dataset.update_candidate_labels(
                            sample_idx.item(),
                            new_cm[i].nonzero(as_tuple=True)[0].tolist()
                        )

            running_loss += loss.item()
            n_batches += 1

        # EMA-smoothed prior update
        self.update_class_priors(dataloader, device)
        return running_loss / max(n_batches, 1)
