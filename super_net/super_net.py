import os

from typing import Tuple, Union

import torch
from torch import nn

from torch.nn import functional as F

from numpy import random


class ConvolutionBlock(nn.Module):
    def __init__(
        self, in_features: int = 1, out_features: int = 32, choice: str = "both"
    ):
        """
        Convolution block builder
        :param in_features: in features for convolution block
        :param out_features: out features for convolution block
        :param choice: one of: conv3x3, conv5x5 or both
        """
        super().__init__()
        if choice == "both":
            self.conv3x3 = nn.Conv2d(
                in_channels=in_features,
                out_channels=out_features,
                kernel_size=3,
                padding=1,
                stride=1,
            )
            self.conv5x5 = nn.Conv2d(
                in_channels=in_features,
                out_channels=out_features,
                kernel_size=5,
                padding=2,
                stride=1,
            )
            self.forward_method = self.forward_both
        elif choice == "conv3x3":
            self.conv3x3 = nn.Conv2d(
                in_channels=in_features,
                out_channels=out_features,
                kernel_size=3,
                padding=1,
                stride=1,
            )
            self.forward_method = self.forward_3x3
        elif choice == "conv5x5":
            self.conv5x5 = nn.Conv2d(
                in_channels=in_features,
                out_channels=out_features,
                kernel_size=5,
                padding=2,
                stride=1,
            )
            self.forward_method = self.forward_5x5
        else:
            raise NotImplementedError(f"{choice} is unknown")

    def forward_both(self, x):
        x3 = self.conv3x3(x)
        x5 = self.conv5x5(x)
        return x3 + x5

    def forward_3x3(self, x):
        return self.conv3x3(x)

    def forward_5x5(self, x):
        return self.conv5x5(x)

    def forward(self, x):
        return self.forward_method(x)


class SuperNet(nn.Module):
    def __init__(
        self,
        in_features: int = 1,
        hidden_features: int = 32,
        out_conv_features: int = 32,
        out_linear_features: int = 64,
        num_classes: int = 10,
        image_size: int = 28,
        conv_1_choice: str = "both",
        conv_2_choice: str = "both",
    ):
        super().__init__()
        self.conv_1_choice = conv_1_choice
        self.conv_2_choice = conv_2_choice
        self.conv_block_1 = ConvolutionBlock(
            in_features=in_features,
            out_features=hidden_features,
            choice=self.conv_1_choice,
        )
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2,)
        self.conv_block_2 = ConvolutionBlock(
            in_features=hidden_features,
            out_features=out_conv_features,
            choice=self.conv_2_choice,
        )
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2)
        linear_in_features = ((image_size // 4) ** 2) * out_conv_features
        self.linear_1 = nn.Linear(
            in_features=linear_in_features, out_features=out_linear_features
        )
        self.linear_2 = nn.Linear(
            in_features=out_linear_features, out_features=num_classes
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.max_pool_1(F.relu(x))
        x = self.conv_block_2(x)
        x = self.max_pool_2(F.relu(x))
        x = self.linear_1(x.view(x.shape[0], -1))
        return F.relu(self.linear_2(x))

    def load_subnet(self, path: str, map_location: str = "cpu"):
        state_dict = torch.load(path, map_location=map_location)
        try:
            self.load_state_dict(state_dict)
            return
        except (KeyError, RuntimeError):
            pass

        if self.conv_1_choice == "conv3x3":
            del state_dict["conv_block_1.conv5x5.weight"]
            del state_dict["conv_block_1.conv5x5.bias"]
        elif self.conv_1_choice == "conv5x5":
            del state_dict["conv_block_1.conv3x3.weight"]
            del state_dict["conv_block_1.conv3x3.bias"]
        elif self.conv_1_choice != "both":
            raise NotImplementedError(
                f"unknown parameter {self.conv_1_choice} for conv_1_choice"
            )

        if self.conv_2_choice == "conv3x3":
            del state_dict["conv_block_2.conv5x5.weight"]
            del state_dict["conv_block_2.conv5x5.bias"]
        elif self.conv_2_choice == "conv5x5":
            del state_dict["conv_block_2.conv3x3.weight"]
            del state_dict["conv_block_2.conv3x3.bias"]
        elif self.conv_2_choice != "both":
            raise NotImplementedError(
                f"unknown parameter {self.conv_2_choice} for conv_2_choice"
            )

        self.load_state_dict(state_dict)


def sample_model(
    in_features: int = 1,
    hidden_features: int = 32,
    out_conv_features: int = 32,
    out_linear_features: int = 64,
    num_classes: int = 10,
    image_size: int = 28,
    conv_1_choice: str = "both",
    conv_2_choice: str = "both",
    model_path: Union[None, str] = None,
    map_location: str = "cpu",
    verbose: bool = False,
) -> SuperNet:
    """
    sample net from super net
    :param in_features: image in channels
    :param hidden_features: out features for first conv block
    :param out_conv_features: out features for second conv block
    :param out_linear_features: out features for first linear block
    :param num_classes: num classes (multiclass classification)
    :param image_size: input image size (height = width)
    :param model_path: path to load model weights (if exists)
    :param map_location: map location for loading weights
    :param conv_1_choice: choice for first conv block. One of: [conv3x3, conv5x5]
    :param conv_2_choice: choice for second conv block. One of: [conv3x3, conv5x5]
    :param verbose: verbose mode
    :return: SuperNet object
    """
    model = SuperNet(
        in_features=in_features,
        hidden_features=hidden_features,
        out_conv_features=out_conv_features,
        out_linear_features=out_linear_features,
        num_classes=num_classes,
        image_size=image_size,
        conv_1_choice=conv_1_choice,
        conv_2_choice=conv_2_choice,
    )
    if model_path is not None and os.path.isfile(model_path):
        if verbose:
            print(f"Loading from {model_path}")
        model.load_subnet(model_path, map_location=map_location)

    return model


def sample_random_net(
    in_features: int = 1,
    hidden_features: int = 32,
    out_conv_features: int = 32,
    out_linear_features: int = 64,
    num_classes: int = 10,
    image_size: int = 28,
    model_path: Union[None, str] = None,
    map_location: str = "cpu",
    verbose: bool = False,
) -> Tuple[SuperNet, str]:
    """
    sample random net from super net
    :param in_features: image in channels
    :param hidden_features: out features for first conv block
    :param out_conv_features: out features for second conv block
    :param out_linear_features: out features for first linear block
    :param num_classes: num classes (multiclass classification)
    :param image_size: input image size (height = width)
    :param model_path: path to load model weights (if exists)
    :param map_location: map location for loading weights
    :param verbose: verbose mode
    :return: SuperNet object, and sampled model name
    """
    conv_1_choice = random.choice(["conv3x3", "conv5x5"])
    conv_2_choice = random.choice(["conv3x3", "conv5x5"])

    if verbose:
        print("*" * 10)
        print(f"First block: {conv_1_choice}")
        print("*" * 5)
        print(f"Second block: {conv_2_choice}")
        print("*" * 10)

    model_name = f"{conv_1_choice}_{conv_2_choice}"
    model = sample_model(
        in_features=in_features,
        hidden_features=hidden_features,
        out_conv_features=out_conv_features,
        out_linear_features=out_linear_features,
        num_classes=num_classes,
        image_size=image_size,
        conv_1_choice=conv_1_choice,
        conv_2_choice=conv_2_choice,
        model_path=model_path,
        map_location=map_location,
    )
    return model, model_name


def save_super_net(
    model: SuperNet, model_path: str, verbose: bool = False,
) -> None:
    """
    Saves model or updates weights
    :param model: model to save
    :param model_path: path to store weights
    :param verbose: verbose mode
    :return:
    """
    if not os.path.isfile(model_path):
        torch.save(model.state_dict(), model_path)
        if verbose:
            print(f"Saved new model in {model_path}")
    else:
        state_dict = torch.load(model_path)
        state_dict.update(model.state_dict())
        torch.save(state_dict, model_path)
        if verbose:
            print(f"Updated new model in {model_path}")
