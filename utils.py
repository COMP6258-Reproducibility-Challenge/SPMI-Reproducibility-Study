import torch
import torch.nn.functional as F

def update_class_priors(outputs, priors):
    probs = F.softmax(outputs, dim=1).detach()
    return 0.9 * priors + 0.1 * probs.mean(dim=0)

def expand_labels(outputs, candidate_sets, priors):
    probs = F.softmax(outputs, dim=1).detach()
    for i in range(len(candidate_sets)):
        for k in range(probs.shape[1]):
            if candidate_sets[i][k] == 0 and probs[i][k] > priors[k]:
                candidate_sets[i][k] = 1
    return candidate_sets

def condense_labels(outputs, candidate_sets, tau=2.0):
    probs = F.softmax(outputs, dim=1).detach()
    for i in range(len(candidate_sets)):
        S = candidate_sets[i].nonzero(as_tuple=False).view(-1)
        if len(S) <= 1:
            continue
        losses = []
        for k in S:
            # 从 S 中剔除 k
            reduced = [label.item() for label in S if label.item() != k.item()]
            if len(reduced) == 0:
                continue
            # 用 reduced 对 probs[i] 索引
            p_full = probs[i][reduced]
            p_detach = probs[i][reduced].detach()
            loss = F.kl_div(torch.log(p_full + 1e-8), p_detach, reduction='sum')
            losses.append((loss.item(), k.item()))
        if len(losses) == 0:
            continue
        min_loss, min_k = min(losses)
        if min_loss < tau:
            candidate_sets[i][min_k] = 0
    return candidate_sets