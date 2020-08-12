from argparse import ArgumentParser

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import safitty

from mnist_dataloader import get_train_dataloader, get_test_dataloader
from super_net import sample_random_net, sample_model, save_super_net
from train_utils import (
    prepare_model,
    train_epoch,
    validate_epoch,
    get_criterion,
)


def seed_all():
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)


def get_configs():
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    args = parser.parse_args()
    return safitty.load(args.config)


def train_super_net(configs):

    assert configs["model_path"] is not None
    model_path = configs["model_path"]
    log_dir = configs.get("tensorboard_log_dir", "./runs")
    writer = SummaryWriter(log_dir)
    number_epochs = configs.get("number_train_epochs", 200)
    train_dataloader = get_train_dataloader(
        **configs["train_dataloader_parameters"]
    )
    test_dataloader = get_test_dataloader(
        **configs["test_dataloader_parameters"]
    )
    criterion = get_criterion(configs)

    model_parameters = configs["model_parameters"] or {}
    # initialize super net with random weights at first
    save_super_net(
        sample_model(model_path=model_path, **model_parameters), model_path
    )

    for epoch in range(number_epochs):
        print(f"Epoch: {epoch + 1}")

        model, model_name = sample_random_net(
            model_path=model_path, **model_parameters
        )
        writer.add_text("model_name", model_name, epoch)
        model, optimizer, _, device = prepare_model(model, configs)
        criterion = criterion.to(device)
        train_epoch(
            model,
            criterion,
            optimizer,
            device,
            train_dataloader,
            epoch,
            writer,
            scheduler=None,
            verbose=True,
        )
        validate_epoch(
            model,
            criterion,
            device,
            test_dataloader,
            epoch,
            writer,
            verbose=True,
        )
        save_super_net(model, model_path=model_path)


if __name__ == "__main__":
    seed_all()
    configs = get_configs()
    train_super_net(configs)
