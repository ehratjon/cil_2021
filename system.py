import torch
from torch import nn

import torchmetrics as tm

import pytorch_lightning as pl

import numpy as np
import matplotlib.pyplot as plt
import random

import torchvision
import torchvision.transforms.functional as F
from torchvision.utils import draw_segmentation_masks

class SemanticSegmentationSystem(pl.LightningModule):
    def __init__(self, model: nn.Module, datamodule: pl.LightningDataModule, model_fix: nn.Module = None, lr: float = 1e-3, batch_size: int = 8):
        super().__init__()
        
        self.model = model
        self.datamodule = datamodule
        
        self.model_fix = model_fix
        
        self.lr = lr
        self.batch_size = batch_size
        
        self.dice_loss = DiceLoss()

    def forward(self, X):
        y_pred = self.model(X.float())
        
        if self.model_fix:
            y_pred_fix = self.model_fix(torch.sigmoid(y_pred))

            return torch.sigmoid(y_pred_fix)
        
        return torch.sigmoid(y_pred)
        
    def training_step(self, batch, batch_idx):
        X, y = batch
        
        X = X.float()
        y = y.float()
        
        y_pred = self.model(X)
       
        loss = self.dice_loss(y_pred, y) + nn.functional.binary_cross_entropy_with_logits(y_pred, y, reduction='mean')
        
        if self.model_fix:
            y_pred_fix = self.model_fix(torch.sigmoid(y_pred))

            loss_fix = self.dice_loss(y_pred_fix, y) + nn.functional.binary_cross_entropy_with_logits(y_pred_fix, y, reduction='mean')

            loss = loss + loss_fix
        
        self.log('training_loss', loss)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        X, y = batch
                
        X = X.float()
        y = y.int()
        
        y_pred = self.model(X)
        y_sig = torch.sigmoid(y_pred)
        
        if self.model_fix:
            y_pred_fix = self.model_fix(y_sig)
            y_sig = torch.sigmoid(y_pred_fix)
       
        accuracy = tm.functional.accuracy(y_sig, y)
        f1 = tm.functional.f1(y_sig, y)
        
        self.log('validation_accuracy', accuracy, prog_bar=True)
        self.log('validation_f1', f1, prog_bar=True)
        
        #show_image(y_sig[0]) 
        
        return accuracy
    
    def test_step(self, batch, batch_idx):
        X, name = batch
        
        X = X.float()

        y_preds = []
        
        for x in X:
            y_pred = self.predict_patches(x)

            y_preds.append(y_pred)
        
        y_preds = torch.stack(y_preds)

        return (y_preds, name)

    def test_epoch_end(self, outputs):
        self.test_results = outputs

    def predict_patches(self, patches, H=608, W=608):
        y_pred_patches = self(patches)

        y_pred = self.restore_image_mask(y_pred_patches, H, W, 80, 4, 4)

        return y_pred
    
    @torch.no_grad()
    def visualize_results(self):
        Xs, ys = next(iter(self.val_dataloader()))
                
        y_preds = torch.sigmoid(self.model(Xs.float().cuda()))
        
        if self.model_fix:
            y_preds = torch.sigmoid(self.model_fix(y_preds))
        
        for y_pred in y_preds:
            show_image(y_pred)
           
    @torch.no_grad()
    def visualize_results_overlay(self, num_images=None):
        Xs, ys = next(iter(self.val_dataloader()))
                
        y_preds = torch.sigmoid(self.model(Xs.float().cuda()))
        
        if self.model_fix:
            y_preds = torch.sigmoid(self.model_fix(y_preds))
        
        imgs_masks_zip = list(zip(Xs, ys))
        seg_imgs_masks = [draw_segmentation_masks(train_pair[0], train_pair[1].bool(), alpha=0.6, colors=['#FF0000']) for train_pair in imgs_masks_zip]
        
        pred_zip = list(zip(seg_imgs_masks, y_preds))
        seg_imgs_pred = [draw_segmentation_masks(train_pair[0], train_pair[1].round().bool(), alpha=0.6, colors=['#00ff00']) for train_pair in pred_zip]
        
        for i, seg_image in enumerate(seg_imgs_pred):
            show_image(seg_image)

    def split_image(self, imgs, kernel_size=240, stride=80):
        if len(imgs.shape) < 4:
            imgs = imgs[None, :, :, :]
        
        B, C, H, W = imgs.shape
        
        patches = imgs.float().unfold(3, kernel_size, stride).unfold(2, kernel_size, stride).permute(0,1,2,3,5,4)
        
        return patches.contiguous().view(patches.shape[0], patches.shape[1], patches.shape[2] * patches.shape[3], patches.shape[4], patches.shape[5]).squeeze(0).transpose(1, 0).byte()
  
    def restore_image(self, patches, H=400, W=400, stride=80):
        B, C, _, _, kernel_size, _ = patches.shape
            
        patches = patches.contiguous().view(B, C, -1, kernel_size*kernel_size)
        patches = patches.permute(0, 1, 3, 2) 
        patches = patches.contiguous().view(B, C*kernel_size*kernel_size, -1)
            
        output = torch.nn.functional.fold(
            patches, output_size=(H, W), kernel_size=kernel_size, stride=stride)
            
        return output

    def restore_image_mask(self, patches, H=400, W=400, stride=80, num_patches_v=3, num_patches_h=3):
        patches = patches.transpose(1, 0).view(patches.shape[1], num_patches_v, num_patches_h, patches.shape[2], patches.shape[3]).unsqueeze(0)    

        B, C, _, _, kernel_size, _ = patches.shape
        
        restored_output = self.restore_image(patches, H, W, stride)
        
        ones = torch.ones((B, C, H, W), device='cuda')
        
        patches_ones = self.split_image(ones, kernel_size, stride).float()
        
        patches_ones = patches_ones.transpose(1, 0).view(patches_ones.shape[1], num_patches_v, num_patches_h, patches_ones.shape[2], patches_ones.shape[3]).unsqueeze(0)    
        
        restored_ones = self.restore_image(patches_ones, H, W, stride)
            
        return restored_output / restored_ones 
            
    def train_dataloader(self):
        return self.datamodule.train_dataloader()

    def val_dataloader(self):
        return self.datamodule.val_dataloader()

    def test_dataloader(self):
        return self.datamodule.test_dataloader()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 10, 2, 1e-6, verbose=2)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'validation_accuracy'
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
    
def show_image(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    
    plt.show()