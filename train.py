import os
import argparse
import logging
from importlib import import_module

import yaml
import torch
from tqdm import tqdm

from src.utils import logging_setup, save_model, AverageMeterSet


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        type=str,
                        default="config/mvc.yaml",
                        help="Path to config file (default: config/mvc.yaml)")
    parser.add_argument("--gpu",
                        type=int,
                        default=0,
                        help="Which gpu to use (default: 0)")
    parser.add_argument("--logging",
                        type=bool,
                        default=True,
                        help='Whether to save logs (default: True)')
    parser.add_argument("--testing-freq",
                        type=int,
                        default=1,
                        help="Frequency of testing in epochs (default: 1) "
                        "(zero means no testing)")
    parser.add_argument(
        "--save-model-path",
        type=str,
        default="model/mvc/model.tar",
        help="Path to save model (default: model/mvc/model.tar)")
    parser.add_argument(
        "--save-model-freq",
        type=int,
        default=10,
        help="Freqency of saving model in epochs (default: 10) "
        "(zero means no saving)")
    args = parser.parse_args()
    return args


def compute_accuracy(outputs, labels):
    """Compute captcha accuracy and per-character accuracy.

    Args:
        outputs: tensor of size [batch_size, num_class, length]
        labels: tensor of size [batch_size, length]

    Returns:
        float: Average captcha accuracy
        float: Averarge accuracy per character
    """
    correct = (outputs.argmax(dim=1) == labels)  # [batch_size, length]
    success = correct.all(dim=1).float().mean().item()
    accuracy = correct.float().mean().item()
    return success, accuracy


def main():
    args = get_args()

    # Load config
    with open(args.config) as fd:
        spec = yaml.load(fd, Loader=yaml.FullLoader)

    # Set device, random seed, logging, and tensorboard
    device = torch.device(f"cuda:{args.gpu}") if torch.cuda.is_available(
    ) else torch.device("cpu")
    torch.manual_seed(spec['seed'])
    writer = logging_setup(args.config) if args.logging else None

    # Prepare data loaders
    module = import_module(f"src.{spec['task']}")
    get_loaders = getattr(module, "get_loaders")
    trainloader, validloader, testloader = get_loaders(spec)

    # Create model, criterion, and optimizer
    ModelClass = getattr(module, spec['model'])
    model = ModelClass(**spec['model_kwargs']).to(device)

    CriterionClass = getattr(module, spec['criterion'])
    criterion = CriterionClass(**spec['criterion_kwargs'])

    optim_module = import_module("torch.optim")
    OptimizerClass = getattr(optim_module, spec['optimizer'])
    optimizer = OptimizerClass(model.parameters(), **spec['optimizer_kwargs'])

    # Main loop
    meters = AverageMeterSet()
    for epoch in range(1, spec['epoch'] + 1):

        # Training
        model.train()
        for data in tqdm(trainloader):

            inputs = data['image'].to(device)
            labels = data['label'].to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            success, accuracy = compute_accuracy(outputs, labels)
            num_data = inputs.size(0)
            meters.update("loss_train", loss.item(), num_data)
            meters.update("success_train", success, num_data)
            meters.update("accuracy_train", accuracy, num_data)

        # Save model
        save_model(args.save_model_path, epoch, model, optimizer)
        if args.save_model_freq > 0 and epoch % args.save_model_freq == 0:
            model_root, model_ext = os.path.splitext(args.save_model_path)
            model_root += f"_{epoch}"
            save_model(model_root + model_ext, epoch, model, optimizer)

        # Validation and testing
        if args.testing_freq > 0 and epoch % args.testing_freq == 0:

            model.eval()
            with torch.no_grad():
                for datatype in ["valid", "test"]:
                    dataloader = locals()[f'{datatype}loader']
                    if dataloader is not None:
                        for data in dataloader:
                            inputs = data['image'].to(device)
                            labels = data['label'].to(device)

                            outputs = model(inputs)
                            loss = criterion(outputs, labels)

                            success, accuracy = compute_accuracy(
                                outputs, labels)
                            num_data = inputs.size(0)
                            meters.update(f"loss_{datatype}", loss.item(),
                                          num_data)
                            meters.update(f"success_{datatype}", success,
                                          num_data)
                            meters.update(f"accuracy_{datatype}", accuracy,
                                          num_data)

        # Printing
        infostr = (f"Epoch {epoch:3d} "
                   f"loss_train {meters['loss_train'].avg:.4f} "
                   f"success_train {meters['success_train'].avg:.4f} "
                   f"accuracy_train {meters['accuracy_train'].avg:.4f} ")
        if args.testing_freq > 0 and epoch % args.testing_freq == 0:
            if validloader is not None:
                infostr += (
                    f"loss_valid {meters['loss_valid'].avg:.4f} "
                    f"success_valid {meters['success_valid'].avg:.4f} "
                    f"accuracy_valid {meters['accuracy_valid'].avg:.4f} ")
            if testloader is not None:
                infostr += (
                    f"loss_test  {meters['loss_test'].avg:.4f} "
                    f"success_test  {meters['success_test'].avg:.4f} "
                    f"accuracy_test  {meters['accuracy_test'].avg:.4f} ")
        print(infostr)

        # Logging
        if args.logging:
            logging.info(infostr)
            loss_dict = {'train': meters['loss_train'].avg}
            success_dict = {'train': meters['success_train'].avg}
            accuracy_dict = {'train': meters['accuracy_train'].avg}
            if args.testing_freq > 0 and epoch % args.testing_freq == 0:
                for datatype in ["valid", "test"]:
                    if locals()[f"{datatype}loader"] is not None:
                        loss_dict[datatype] = meters[f'loss_{datatype}'].avg
                        success_dict[datatype] = meters[
                            f'success_{datatype}'].avg
                        accuracy_dict[datatype] = meters[
                            f'accuracy_{datatype}'].avg
            writer.add_scalars("loss", loss_dict, epoch)
            writer.add_scalars("success", success_dict, epoch)
            writer.add_scalars("accuracy", accuracy_dict, epoch)

        # Reset meters for each epoch
        meters.reset()

    # Clean up
    if args.logging:
        writer.close()


if __name__ == "__main__":
    main()
