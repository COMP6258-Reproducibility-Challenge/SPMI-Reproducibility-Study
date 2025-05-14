# debug_spmi.py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from spmi import SPMI
from model import LeNet

class TinyDataset(Dataset):
    def __init__(self, num_samples=8, img_shape=(1,28,28), num_classes=2):
        self.num_classes = num_classes
        self.imgs = torch.randn(num_samples, *img_shape)
        # True labels 0…num_classes-1
        self.targets = torch.randint(0, num_classes, (num_samples,))
        # First half labeled
        self.is_labeled = torch.tensor([1]*(num_samples//2) + [0]*(num_samples - num_samples//2))
        # Candidate masks -> for labeled, start with only true label; unlabeled start empty
        self.cand = torch.zeros(num_samples, num_classes, dtype=torch.int)
        for i in range(num_samples):
            if self.is_labeled[i]==1:
                self.cand[i, self.targets[i]] = 1

    def __len__(self): return len(self.imgs)
    def __getitem__(self, i):
        return (self.imgs[i],
                self.cand[i].float(),
                self.targets[i],
                i,                     # global index
                self.is_labeled[i])

    
    def update_candidate_labels(self, idx, new_candidates):
        """
        
                        or a Python list of class-indices.
        We convert it into a binary mask and write into self.cand[idx].
        """
        # If they passed you a full mask tensor then turn it into indices
        if isinstance(new_candidates, torch.Tensor):
            # either a Tensor mask of shape [num_classes]
            new_candidates = new_candidates.nonzero(as_tuple=True)[0].tolist()

        # Zero out the old mask
        self.cand[idx].zero_()

        # Set 1s at each candidate index
        for k in new_candidates:
            self.cand[idx, k] = 1

        # if we accidentally clear everything then fall back to the true label
        if self.cand[idx].sum() == 0:
            true_label = self.targets[idx].item()  # or .numpy()[0]
            self.cand[idx, true_label] = 1

# Instantiate
ds = TinyDataset()
loader = DataLoader(ds, batch_size=4, shuffle=True)
device = torch.device('cpu')
model = LeNet(num_classes=2).to(device)
spmi = SPMI(model, num_classes=2, tau=0.5, unlabeled_tau=0.5)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Warmup epoch:
print("Warmup epoch")
loss = spmi.train_epoch(loader, optimizer, device,
                        epoch=0, warmup_epochs=1,
                        original_candidate_masks=None)
print("Warmup loss:", loss)

# Initialize unlabeled:
spmi.initialize_unlabeled(loader, device)
print("After init, candidate masks:\n", ds.cand)

# 3) Post-warmup epoch:
print("\n Post warmup epoch")
loss = spmi.train_epoch(loader, optimizer, device,
                        epoch=1, warmup_epochs=1,
                        original_candidate_masks=ds.cand.clone())
print("Post warmup loss:", loss)
print("Final candidate masks:\n", ds.cand)
