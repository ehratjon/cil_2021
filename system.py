import torch
from torch import nn

import torchmetrics as tm

import pytorch_lightning as pl

class SemanticSegmentationSystem(pl.LightningModule):
    def __init__(self, model: nn.Module, datamodule: pl.LightningDataModule, lr: float = 1e-3, batch_size: int = 80):
        super().__init__()
        
        self.model = model
        self.datamodule = datamodule
        
        self.lr = lr
        self.batch_size = batch_size
        
        self.dice_loss = DiceLoss()

    def training_step(self, batch, batch_idx):
        X, y = batch
        
        X = X.float()
        y = y.float()
        
        y_pred = self.model(X)
       
        #loss = sigmoid_focal_loss(y_pred, y, reduction='mean')
        #loss = nn.functional.binary_cross_entropy_with_logits(y_pred, y, reduction='mean')
        #loss = sigmoid_focal_loss(y_pred, y, reduction='mean') + nn.functional.binary_cross_entropy_with_logits(y_pred, y, reduction='mean')
        loss = self.dice_loss(y_pred, y) + nn.functional.binary_cross_entropy_with_logits(y_pred, y, reduction='mean')
        
        self.log('training_loss', loss)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        X, y = batch
                
        X = X.float()
        y = y.int()
        
        y_pred = self.model(X)
        y_sig = torch.sigmoid(y_pred)
       
        metric = tm.functional.accuracy(y_sig, y, average='samples')
        
        self.log('validation_metric', metric)
        
        return metric
    
    def test_step(self, batch, batch_idx):
        X, _ = batch
        
        return self.model(X)
    
    @torch.no_grad()
    def visualize_results(self):
        Xs, ys = next(iter(self.val_dataloader()))
                
        y_preds = torch.sigmoid(self.model(X.float().cuda()))
        
        for y_pred in y_preds:
            show_image(y_pred)
           
    @torch.no_grad()
    def visualize_results_overlay(self, num_images=None):
        Xs, ys = next(iter(self.val_dataloader()))
                
        y_preds = torch.sigmoid(self.model(Xs.float().cuda()))
        
        imgs_masks_zip = list(zip(Xs, y))
        seg_imgs_masks = [draw_segmentation_masks(train_pair[0], train_pair[1].bool(), colors=['#FF0000']) for train_pair in imgs_masks_zip]
        
        pred_zip = list(zip(seg_imgs_masks, y_preds))
        seg_imgs_pred = [draw_segmentation_masks(train_pair[0], train_pair[1].round().bool(), colors=['#00ff00']) for train_pair in pred_zip]
        
        for i, seg_image in enumerate(seg_imgs_pred):
            show_image(seg_image)
            
            if i >= num_images:
                return
            
    def train_dataloader(self):
        return self.datamodule.train_dataloader()

    def val_dataloader(self):
        return self.datamodule.val_dataloader()

    def test_dataloader(self):
        return self.datamodule.test_dataloader()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, verbose=2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 10, 2, 1e-6, verbose=2)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'validation_metric'
        }

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice