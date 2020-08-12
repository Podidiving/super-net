from argparse import ArgumentParser

import os
import safitty
import torch
import numpy as np

from utils.mnist_dataloader import get_test_dataloader
from super_net import sample_model
from utils.train_utils import get_criterion, prepare_model, validate_epoch


def seed_all():
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)


def get_configs():
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-d", "--dump", type=str, required=False, default=None, help="File to dump results to")
    args = parser.parse_args()
    return safitty.load(args.config), args.dump


if __name__ == "__main__":
    seed_all()
    configs, dump = get_configs()
    model_parameters = configs["model_parameters"] or {}
    model_path = configs["model_path"]
    assert os.path.isfile(model_path)
    test_dataloader = get_test_dataloader(
        **configs["test_dataloader_parameters"]
    )
    criterion = get_criterion(configs)
    for conv_1_choice in ["conv3x3", "conv5x5"]:
        for conv_2_choice in ["conv3x3", "conv5x5"]:
            model = sample_model(
                model_path=model_path,
                conv_1_choice=conv_1_choice,
                conv_2_choice=conv_2_choice,
                **model_parameters,
            )
            model_name = f"{conv_1_choice}_{conv_2_choice}"
            model, optimizer, _, device = prepare_model(model, configs)
            averaged_val_loss, averaged_val_acc = validate_epoch(
                model, criterion, device, test_dataloader, None, None, True
            )
            print(f"{model_name} : {averaged_val_acc}, {averaged_val_loss}")
            if dump is not None:
                with open(dump, "a") as file:
                    file.write(f"{model_name} : {averaged_val_acc}, {averaged_val_loss}\n")
