from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.tversky import TverskyLoss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
import numpy as np

class nnUNetTrainerTverskyLoss(nnUNetTrainer):
    def _build_loss(self):
        loss = TverskyLoss(alpha=0.3, beta=0.7, smooth=1e-5)

        deep_supervision_scales = self._get_deep_supervision_scales()
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
        weights = weights / weights.sum()
        loss = DeepSupervisionWrapper(loss, weights)
        return loss
