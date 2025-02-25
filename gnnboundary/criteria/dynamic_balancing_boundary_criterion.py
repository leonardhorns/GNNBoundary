import torch
from torch import nn
import torch.nn.functional as F


class DynamicBalancingBoundaryCriterion(nn.Module):
    def __init__(self, classes, alpha=1, beta=1):
        super().__init__()
        self.classes = classes
        self.alpha = alpha
        self.beta = beta

    def forward(self, logits):
        assert len(logits.shape) == 2
        probs = F.softmax(logits, dim=-1).detach()
        mask = torch.zeros_like(logits).bool()
        mask[:, self.classes] = True
        score = logits * probs ** 2
        notmax = probs < probs.max(dim=1, keepdim=True).values
        score[mask] = logits[mask] * (1 - probs[mask]) ** 2 * notmax[mask]
        return (self.beta * score[~mask].sum(dim=-1) - self.alpha * score[mask].sum(dim=-1)).mean()
