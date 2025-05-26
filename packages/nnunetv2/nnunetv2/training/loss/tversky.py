import torch
import torch.nn as nn
import torch.nn.functional as F

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1.0):
        """
        alpha: weight for false positives
        beta: weight for false negatives
        smooth: small constant to avoid division by zero
        """
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):
        # 1) Remove channel dim if targets is [B,1,D,H,W]
        if targets.dim() == inputs.dim():
            targets = targets[:, 0, ...]  # now [B, D, H, W]

        # 2) Convert logits to probabilities
        probs = F.softmax(inputs, dim=1)  # [B, C, D, H, W]
        B, C, D, H, W = probs.shape

        # 3) One-hot encode targets to shape [B, C, D, H, W]
        t_onehot = F.one_hot(targets.long(), num_classes=C)      # [B, D, H, W, C]
        t_onehot = t_onehot.permute(0, 4, 1, 2, 3).float()        # [B, C, D, H, W]

        # 4) Flatten per sample/class
        p_flat = probs.reshape(B, C, -1)      # [B, C, D*H*W]
        t_flat = t_onehot.reshape(B, C, -1)   # [B, C, D*H*W]

        # 5) Compute per‐class true pos / false pos / false neg
        TP = (p_flat * t_flat).sum(-1)                    # [B, C]
        FP = (p_flat * (1 - t_flat)).sum(-1)              # [B, C]
        FN = ((1 - p_flat) * t_flat).sum(-1)              # [B, C]

        # 6) Tversky index per class
        tversky = (TP + self.smooth) / (
            TP + self.alpha * FP + self.beta * FN + self.smooth
        )  # [B, C]

        # 7) Loss = 1 - index, averaged over batch and classes
        loss = 1.0 - tversky
        return loss.mean()
