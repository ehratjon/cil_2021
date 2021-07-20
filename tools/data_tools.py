import os
import torch
from PIL import Image
import torchvision.transforms.functional as F


def store_tensor_as_image(tensor, path):
    root, ext = os.path.splitext(path)
    if not ext:
        ext = '.png'
    path = root + ext

    image = tensor_to_image(tensor)
    image.save(path)


def tensor_to_image(tensor):
    image_as_array = tensor.to(torch.uint8).numpy()
    if(len(image_as_array) == 3):
        image_as_array = image_as_array.transpose((1, 2, 0))
    image = Image.fromarray(image_as_array)
    return image


def image_to_tensor(image, float=True):
    if(len(image) == 3):
        image = image.transpose((2, 0, 1))
    image = torch.from_numpy(image)
    if(float):
        image = image.float()
    return image


# from mike
# TODO: change from matplotlib to store file
"""
def show_image(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
"""

# from mike to change color (as not to overfit model to color paletter)
# TODO: fix
"""
def change_color(X, y):
    Xs, ys = next(iter(system.val_dataloader()))

    y_preds = torch.sigmoid(model(Xs.float()))

    imgs_masks_zip = list(zip(Xs, y))
    seg_imgs_masks = [draw_segmentation_masks(train_pair[0], train_pair[1].bool(), colors=['#FF0000']) for train_pair in imgs_masks_zip]

    pred_zip = list(zip(seg_imgs_masks, y_preds))
    seg_imgs_pred = [draw_segmentation_masks(train_pair[0], train_pair[1].round().bool(), colors=['#00ff00']) for train_pair in pred_zip]

    for i, seg_image in enumerate(seg_imgs_pred):
        show_image(seg_image)
"""