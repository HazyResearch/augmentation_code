import copy
from collections import namedtuple

import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageFilter, ImageEnhance

# An augmentation object consists of its name, the transform functions of type
# torchvision.transforms, and the resulting augmented dataset of type
# torch.utils.data.Dataset.
Augmentation = namedtuple('Augmentation', ['name', 'transforms', 'dataset'])


def copy_with_new_transform(dataset, transform):
    """A copy of @dataset with its transform set to @transform.
    Will work for datasets from torchvision, e.g., MNIST, CIFAR10, etc. Probably
    won't work for a generic dataset.
    """
    new_dataset = copy.copy(dataset)
    new_dataset.transform = transform
    return new_dataset


def augment_transforms(augmentations, base_transform, add_id_transform=True):
    """Construct a new transform that stack all the augmentations.
    Parameters:
        augmentations: list of transforms (e.g. image rotations)
        base_transform: transform to be applied after augmentation (e.g. ToTensor)
        add_id_transform: whether to include the original image (i.e. identity transform) in the new transform.
    Return:
        a new transform that takes in a data point and applies all the
        augmentations, then stack the result.
    """
    if add_id_transform:
        fn = lambda x: torch.stack([base_transform(x)] + [base_transform(aug(x))
                                                          for aug in augmentations])
    else:
        fn = lambda x: torch.stack([base_transform(aug(x)) for aug in augmentations])
    return transforms.Lambda(fn)


def rotation(base_dataset, base_transform, angles=range(-15, 16, 2)):
    """Rotations, e.g. between -15 and 15 degrees
    """
    rotations = [transforms.RandomRotation((angle, angle)) for angle in angles]
    aug_dataset = copy_with_new_transform(base_dataset,
                                          augment_transforms(rotations, base_transform))
    return Augmentation('rotation', rotations, aug_dataset)


def resized_crop(base_dataset, base_transform, size=28, scale=(0.64, 1.0), n_random_samples=31):
    """Random crop (with resize)
    """
    random_resized_crops = [transforms.RandomResizedCrop(size, scale=scale) for _ in range(n_random_samples)]
    aug_dataset = copy_with_new_transform(base_dataset,
                                          augment_transforms(random_resized_crops, base_transform))
    return Augmentation('crop', random_resized_crops, aug_dataset)


def blur(base_dataset, base_transform, radii=np.linspace(0.05, 1.0, 20)):
    """Random Gaussian blur
    """
    def gaussian_blur_fn(radius):
        return transforms.Lambda(lambda img: img.filter(ImageFilter.GaussianBlur(radius)))

    blurs = [gaussian_blur_fn(radius) for radius in radii]
    aug_dataset = copy_with_new_transform(base_dataset,
                                          augment_transforms(blurs, base_transform))
    return Augmentation('blur', blurs, aug_dataset)


def rotation_crop_blur(base_dataset, base_transform, angles=range(-15, 16, 2),
                       size=28, scale=(0.64, 1.0), n_random_samples=31,
                       radii=np.linspace(0.05, 1.0, 20)):
    """All 3: rotations, random crops, and blurs
    """
    rotations = rotation(base_dataset, base_transform, angles).transforms
    random_resized_crops = resized_crop(base_dataset, base_transform, size, scale, n_random_samples).transforms
    blurs = blur(base_dataset, base_transform, radii).transforms
    aug_dataset = copy_with_new_transform(base_dataset,
                                          augment_transforms(rotations + random_resized_crops + blurs, base_transform))
    return Augmentation('rotation_crop_blur', blurs, aug_dataset)


def hflip(base_dataset, base_transform):
    """Horizontal flip
    """
    flip = [transforms.Lambda(lambda img: img.transpose(Image.FLIP_LEFT_RIGHT))]
    aug_dataset = copy_with_new_transform(base_dataset,
                                          augment_transforms(flip, base_transform))
    return Augmentation('hflip', flip, aug_dataset)


def hflip_vflip(base_dataset, base_transform):
    """Both horizontal and vertical flips
    """
    allflips = [transforms.Lambda(lambda img: img.transpose(Image.FLIP_LEFT_RIGHT)),
                transforms.Lambda(lambda img: img.transpose(Image.FLIP_TOP_BOTTOM)),
                transforms.Lambda(lambda img: img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM))]
    aug_dataset = copy_with_new_transform(base_dataset,
                                          augment_transforms(allflips, base_transform))
    return Augmentation('hflip_vflip', allflips, aug_dataset)


def brightness(base_dataset, base_transform, brightness_factors=np.linspace(1 - 0.25, 1 + 0.25, 11)):
    """Random brightness adjustment
    """
    def brightness_fn(brightness_factor):
        return transforms.Lambda(lambda img: ImageEnhance.Brightness(img).enhance(brightness_factor))

    brightness_transforms = [brightness_fn(factor) for factor in brightness_factors]
    aug_dataset = copy_with_new_transform(base_dataset,
                                          augment_transforms(brightness_transforms, base_transform))
    return Augmentation('brightness', brightness_transforms, aug_dataset)


def contrast(base_dataset, base_transform, contrast_factors=np.linspace(1 - 0.35, 1 + 0.35, 11)):
    """Random contrast adjustment
    """
    def contrast_fn(contrast_factor):
        return transforms.Lambda(lambda img: ImageEnhance.Contrast(img).enhance(contrast_factor))

    contrast_transforms = [contrast_fn(factor) for factor in contrast_factors]
    aug_dataset = copy_with_new_transform(base_dataset,
                                          augment_transforms(contrast_transforms, base_transform))
    return Augmentation('contrast', contrast_transforms, aug_dataset)
