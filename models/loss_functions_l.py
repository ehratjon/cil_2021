from torch.nn import functional
import pytorch_lightning as pl

class LightningCrossEntropyLoss(pl.LightningModule):
    
    def cross_entropy_loss(self, logits, labels):
        return functional.nll_loss(logits, labels)