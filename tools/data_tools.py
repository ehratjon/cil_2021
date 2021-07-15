import torch
from PIL import Image


def float_tensor_to_image(tensor):
    image_as_array = tensor.to(torch.uint8).numpy()
    if(len(image_as_array) == 3):
        image_as_array = image_as_array.transpose((1, 2, 0))
    image = Image.fromarray(image_as_array)
    return image


def image_to_float_tensor(image):
    if(len(image) == 3):
        image = image.transpose((2, 0, 1))
    image = torch.from_numpy(image).float()
    return image