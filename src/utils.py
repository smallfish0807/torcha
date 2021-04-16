import os
import shutil
import logging

import torch
from torch.utils.tensorboard import SummaryWriter


def logging_setup(config_path, append=False):
    # If configuration file is "config/xxx.yaml", save logging to directory
    # "log/xxx/"
    paths = os.path.normpath(config_path).split(os.sep)
    if paths[0] == "config":
        del paths[0]
    dir_path = os.path.splitext(os.path.join("log", *paths))[0]
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    shutil.copy2(config_path, os.path.join(dir_path, "config.yaml"))

    filename = os.path.join(dir_path, "logger")
    filemode = 'a' if append else 'w'
    logging.basicConfig(filename=filename,
                        filemode=filemode,
                        format="%(levelname)s: %(message)s",
                        level=logging.DEBUG)
    return SummaryWriter(log_dir=dir_path)


def save_model(path, epoch, model, optimizer):
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, path)


def load_model(path, device, model, optimizer=None):
    map_location = torch.device('cpu') if device == torch.device(
        'cpu') else None
    checkpoint = torch.load(path, map_location=map_location)

    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']

    return model, optimizer, epoch


class AverageMeterSet:
    def __init__(self):
        self.meters = {}

    def __getitem__(self, key):
        return self.meters[key]

    def update(self, name, value, n=1):
        if name not in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def values(self, postfix=''):
        return {
            name + postfix: meter.val
            for name, meter in self.meters.items()
        }

    def averages(self, postfix='/avg'):
        return {
            name + postfix: meter.avg
            for name, meter in self.meters.items()
        }

    def sums(self, postfix='/sum'):
        return {
            name + postfix: meter.sum
            for name, meter in self.meters.items()
        }

    def counts(self, postfix='/count'):
        return {
            name + postfix: meter.count
            for name, meter in self.meters.items()
        }


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, format):
        return "{self.val:{format}} ({self.avg:{format}})".format(
            self=self, format=format)
