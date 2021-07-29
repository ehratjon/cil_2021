from data import RoadSatelliteModule
from system import SemanticSegmentationSystem

from models import *

import numpy as np
import matplotlib.pyplot as plt
import random

import torchvision
import torchvision.transforms.functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

import torch
from torchinfo import summary

import regex as re


batch_size = 8
num_workers = 8
pl.seed_everything(7, workers=True)

road_data = RoadSatelliteModule(num_workers=num_workers, batch_size=batch_size)

model = UNetSpatial(1, 3)
model_name = str(model).partition('(')[0]

system = SemanticSegmentationSystem(model, road_data)
#system = SemanticSegmentationSystem.load_from_checkpoint(f'./lightning_logs/{model_name}.ckpt', model=model, datamodule=road_data)

if torch.cuda.is_available():
    gpu_count = -1
    gpu_auto_select = True
    print("GPUs detected.")
    print("There should be ", torch.cuda.device_count(), " GPUs available.")
else:
    gpu_count = 0
    gpu_auto_select = False
    print("No GPU detected.")
    print("Working with CPU")

checkpoint_callback = ModelCheckpoint(
    monitor='validation_accuracy',
    dirpath='./lightning_logs',
    filename=model_name,
    save_top_k=1,
    verbose=2,
    mode='max'
)

early_stop_callback = EarlyStopping(
   monitor='validation_accuracy',
   patience=60,
   verbose=1,
   mode='max'
)

trainer = pl.Trainer(
    #fast_dev_run=True,
    gpus=gpu_count,
    auto_select_gpus=gpu_auto_select,
    stochastic_weight_avg=True,
    benchmark=True,
    callbacks=[early_stop_callback, checkpoint_callback]
)

if gpu_count != 0:
    model.cuda()


trainer.test()

foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch

# assign a label to a patch
def patch_to_label(patch):
    df = np.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0
    
def mask_to_patched_mask(image):
    patched_image = image.squeeze().detach().clone()
    image = np.asarray(image.squeeze())
    patch_size = 16
    for j in range(0, image.shape[1], patch_size):
        for i in range(0, image.shape[0], patch_size):
            patch = image[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch)
            patched_image[i:i + patch_size, j:j + patch_size] = label
    return patched_image

def mask_to_submission_strings(im, name):
    """Reads a single image and outputs the strings that should go into the submission file"""
    img_number = int(re.search(r"\d+", name).group(0))
    #im = mpimg.imread(image_filename) 
    # image is gray scale therefore size MxN with imread 
    patch_size = 16
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch)
            yield("{:03d}_{}_{},{}".format(img_number, j, i, label))


def masks_to_submission(submission_filename, *images):
    """Converts images into a submission file"""
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for imgs, fn in images[0:]:
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(imgs, fn))


batches = system.test_results

submission_filename = model_name + '_predictions.csv'
pred_counter = 0

with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        
        for mask, name in system.test_results:
            predicted_mask = np.asarray(mask.cpu().squeeze())

            ids = mask_to_submission_strings(predicted_mask, name)
            f.writelines('{}\n'.format('\n'.join(ids)))

            pred_counter += 1
