import torch
import torch.nn as nn

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
        # If dealing with logits, use sigmoid (or softmax for multi-class)
        inputs = torch.sigmoid(inputs)
        
        # Flatten tensors to compute confusion matrix elements over the mini-batch
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        true_pos = (inputs * targets).sum()
        false_neg = (targets * (1 - inputs)).sum()
        false_pos = ((1 - targets) * inputs).sum()
        
        tversky_index = (true_pos + self.smooth) / (true_pos + self.alpha * false_pos + self.beta * false_neg + self.smooth)
        loss = 1 - tversky_index
        return loss