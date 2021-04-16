import os

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

CrossEntropyLoss = nn.CrossEntropyLoss


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class MvcCNN(nn.Module):
    def __init__(self, input_shape, num_class, length):
        super().__init__()
        self.input_shape = input_shape
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32,
                      kernel_size=4), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4),
            nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4),
            nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2), nn.Flatten())
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=self.conv_layers_output_shape(),
                      out_features=512), nn.ReLU(),
            nn.Linear(in_features=512, out_features=num_class * length),
            Lambda(lambda x: x.view(-1, num_class, length)))

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

    def conv_layers_output_shape(self):
        return self.conv_layers(torch.zeros(1, *self.input_shape)).size(1)


def get_loaders(spec):
    """Get data loaders for training, validation, and testing.

    Transform the image to tensor of shape [C, H, W] and normalize it.
    Encode the label to integer.

    Returns:
        torch.utils.data.DataLoader: data loader for training
        torch.utils.data.DataLoader or None: data loader for validation
        torch.utils.data.DataLoader or None: data loader for testing
    """
    transform_all = transforms.Compose([
        SampleTransform(transforms.ToTensor()),
        SampleTransform(transforms.Normalize((0.5, 0.5, 0.5),
                                             (0.5, 0.5, 0.5))),
        SampleTransform(LabelEncoding(spec['chars']), key='label')
    ])
    trainset = MvcDataset(spec['label_file_train'], spec['image_dir_train'],
                          transform_all)
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=spec['batch_size'],
                                              shuffle=True,
                                              num_workers=2)
    if 'label_file_valid' in spec and 'image_dir_valid' in spec:
        validset = MvcDataset(spec['label_file_valid'],
                              spec['image_dir_valid'], transform_all)
        validloader = torch.utils.data.DataLoader(
            validset,
            batch_size=spec['batch_size'],
            shuffle=False,
            num_workers=2)
    else:
        validloader = None
    if 'label_file_test' in spec and 'image_dir_test' in spec:
        testset = MvcDataset(spec['label_file_test'], spec['image_dir_test'],
                             transform_all)
        testloader = torch.utils.data.DataLoader(testset,
                                                 batch_size=spec['batch_size'],
                                                 shuffle=False,
                                                 num_workers=2)
    else:
        testloader = None
    return trainloader, validloader, testloader


class MvcDataset(Dataset):
    """Mvc captcha dataset"""
    def __init__(self, label_file, image_dir, transform=None):
        """
        Args:
            label_file (str): path to the file that contains labels
            image_dir (str): directory of all images
            transform (callable, optional): optional transform to be applied on
                a sample
        """
        self.labels = pd.read_csv(label_file, header=None)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        """
        Returns:
            int: number of data samples
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Returns
            dict{'image': PIL Image object, 'label': str}:
                If transform is None, image is of shape [h, w, 3],
                and label is a string that contains four characters.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Read image file "{image_dir}/{idx}.gif"
        image = Image.open(os.path.join(self.image_dir,
                                        f"{idx}.gif")).convert("RGB")

        sample = {'image': image, 'label': self.labels.iloc[idx, 0]}

        if self.transform:
            sample = self.transform(sample)

        return sample


class SampleTransform(object):
    """Transform wrapper for MvcDataset.

    Can be used to apply transform to sample[key]
    """
    def __init__(self, transform, key='image'):
        self.transform = transform
        if key not in ['image', 'label']:
            raise NotImplementedError
        self.key = key

    def __call__(self, sample):
        if self.key == 'image':
            return {
                'image': self.transform(sample['image']),
                'label': sample['label']
            }
        else:
            return {
                'image': sample['image'],
                'label': self.transform(sample['label'])
            }


class LabelEncoding(object):
    def __init__(self, chars):
        self.char2num = {char: ind for ind, char in enumerate(chars)}

    def __call__(self, label):
        """
        Args:
            label (str): label string

        Returns:
            torch.tensor([length]): integers that represent the characters
        """
        X = [self.char2num[char] for char in label]
        return torch.tensor(X)


class LabelDecoding(object):
    def __init__(self, chars):
        self.chars = chars

    def __call__(self, nums):
        """
        Args:
            nums (torch.tensor([length])): integers

        Returns:
            str: label string
        """
        label = ""
        for num in nums:
            label += self.chars[num.item()]
        return label
