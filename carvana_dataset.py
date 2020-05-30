import os

import torch.utils.data as dt
import torch

from PIL import Image
import torchvision.transforms as transforms
from albumentations import Resize, RandomSizedCrop, RandomCrop, Compose, VerticalFlip, HorizontalFlip, RandomBrightnessContrast, RandomRotate90, Rotate, Normalize
import numpy as np

to_img = transforms.ToPILImage()

# Augmentations
aug_train_both = Compose(
    [Resize(224, 224),
     HorizontalFlip(p=0.5),
     Rotate((-30., 30.), p=0.5)])
aug_train_image = Compose(
    [Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

aug_test_both = Compose([Resize(224, 224)])
aug_test_image = Compose(
    [Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(
            self.mean, self.std)


class CarvanaDataset(dt.Dataset):
    """ 
    Carvana features dataset.  
    Override torch Dataset class to implements reading from h5 files
    """
    def __init__(self, data_path, mask_path, input_size=224, is_train=True):
        """
        Args:
            data_path (string): Path to the images data files.
            mask_path (string): Path were images masks are placed
        """

        self.is_train = is_train

        self.files = os.listdir(data_path)
        self.files.sort()

        self.mask_files = os.listdir(mask_path)
        self.mask_files.sort()

        self.data_path = data_path
        self.mask_path = mask_path

        assert (len(self.files) == len(self.mask_files))
        self.input_size = input_size

    def __len__(self):
        return len(self.files)

    def pil_load(self, path, is_input=True):
        # open path as file to avoid ResourceWarning
        # (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as infile:
            with Image.open(infile) as img:
                if is_input:
                    return img.convert('RGB')
                return img.convert('1')

    def pil_save(self, tensor, img_path):
        image = to_img(tensor)
        image.save(img_path, 'PNG')

    def __getitem__(self, idx):
        file_name = os.path.join(self.data_path, self.files[idx])
        mask_name = os.path.join(self.mask_path, self.mask_files[idx])

        if os.path.exists(file_name) == False:
            raise Exception(f"Missing file with name {file_name} in dataset")

        input = self.pil_load(file_name)
        target = self.pil_load(mask_name, False)

        preprocess_both = aug_train_both if self.is_train else aug_test_both
        res = preprocess_both(image=np.array(input).astype(np.float32),
                              mask=np.array(target).astype(np.float32))
        input = res['image'] / 255.
        target = res['mask'] / 255.

        preprocess_image = aug_train_image if self.is_train else aug_test_image
        res = preprocess_image(image=input)
        input = res['image']

        input = np.transpose(input, (2, 0, 1)).astype(np.float32)

        input = torch.FloatTensor(input)
        target = torch.FloatTensor(target)

        target = target.unsqueeze(0)
        target[torch.gt(target, 0)] = 1

        return input, target