import sys
sys.path.append("../cil_data")

import torch
from sklearn.metrics import f1_score

import mask_to_submission

patch_size = 16

# loss function as described in post: https://piazza.com/class/kl2btmrjb735te?cid=77
def sklearn_f1_score(y_pred, y_true):
    return f1_score(y_true, y_pred, average="samples")

# subroutine of mean_f_score, computes error for one sample
def mean_f_score_subroutine(y_pred, y_true):
    number_of_correct_patches = 0
    number_of_patches = 0
    for j in range(0, y_pred.shape[1], patch_size):
        for i in range(0, y_pred.shape[0], patch_size):
            y_pred_patch = y_pred[i:i + patch_size, j:j + patch_size]
            y_pred_label = mask_to_submission.patch_to_label(y_pred_patch)
            y_true_patch = y_true[i:i + patch_size, j:j + patch_size]
            y_true_label = mask_to_submission.patch_to_label(y_true_patch)

            if(y_pred_label == y_true_label): number_of_correct_patches += 1
            number_of_patches += 1
    return number_of_correct_patches / number_of_patches

"""
since f1 score does not seem to work quite right, we implement our own
inspired from mask_to_submission.py
Note: It seems that for loss functions to fully work, all operations need 
to be done on tensors
"""
def mean_f_score(y_pred, y_true, average=torch.mean):
    assert y_pred.shape == y_true.shape

    # we will need both to be of type float to compute the mean
    y_true = y_true.float()
    
    # without batching y_pred will have 2 dimensions, else 3 and we need
    # to accomodate for that
    if(len(y_pred.shape) == 2):
        return mean_f_score_subroutine(y_pred, y_true)
    else:
        batch_size = y_pred.shape[0]
        scores = []
        for i in range(batch_size):
            scores.append(mean_f_score_subroutine(y_pred[i], y_true[i]))
        return average(torch.tensor(scores, requires_grad=True))
    