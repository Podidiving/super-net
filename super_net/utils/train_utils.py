from typing import Tuple, Union
from tqdm import tqdm
import torch
from torch import nn


def get_criterion(configs):

    assert type(configs["criterion_name"]) is str
    criterion_parameters = configs.get("criterion_parameters", None) or {}
    criterion = torch.nn.__dict__[configs["criterion_name"]](
        **criterion_parameters
    )
    return criterion


def prepare_model(
    model: nn.Module, configs: dict, use_multi_gpu_if_available: bool = False
) -> Tuple[
    "nn.Module", "torch.optim", Union["torch.optim.lr_scheduler", None], str
]:

    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1 and use_multi_gpu_if_available:
            model = nn.DataParallel(model.cuda())
        else:
            model = model.cuda()
    else:
        device = "cpu"
    model = model.train()
    assert configs.get("optimizer_name", None) is not None
    assert configs.get("optimizer_parameters", None) is not None
    optimizer = torch.optim.__dict__[configs["optimizer_name"]](
        model.parameters(), **configs["optimizer_parameters"]
    )
    # lr scheduler seems quite useless, so, I don't use it in experiments
    lr_scheduler = None
    if configs.get("lr_scheduler_name", None) is not None:
        assert configs.get("lr_scheduler_parameters", None) is not None
        lr_scheduler = torch.optim.lr_scheduler.__dict__[
            configs["lr_scheduler_name"]
        ](optimizer, **configs["lr_scheduler_parameters"])

    return model, optimizer, lr_scheduler, device


def train_epoch(
    model: nn.Module,
    criterion,
    optimizer,
    device,
    dataloader,
    epoch_number=None,
    writer=None,
    scheduler=None,
    verbose=True,
):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    dataloader_len = 0
    if verbose:
        dataloader = tqdm(dataloader)

    train_loss = 0.0
    for images, targets in dataloader:
        dataloader_len += targets.shape[0]
        optimizer.zero_grad()
        images = images.to(device)
        targets = targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.detach().cpu().item()
    averaged_train_loss = train_loss / dataloader_len

    if scheduler is not None:
        scheduler.step()
    if writer is not None:
        assert type(epoch_number) is int
        writer.add_scalar(
            "averaged_loss/train", averaged_train_loss, epoch_number
        )

    return averaged_train_loss


def validate_epoch(
    model, criterion, device, dataloader, epoch_number, writer, verbose
):
    model = model.eval()
    dataloader_len = 0
    if verbose:
        dataloader = tqdm(dataloader)

    val_loss = 0.0
    val_acc = 0.0
    with torch.no_grad():
        for images, targets in dataloader:
            dataloader_len += targets.shape[0]
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            val_loss += loss.cpu().item()
            val_acc += (
                torch.sum(outputs.argmax(1) == targets).detach().cpu().item()
            )

    averaged_val_loss = val_loss / dataloader_len
    averaged_val_acc = val_acc / dataloader_len

    if writer is not None:
        assert type(epoch_number) is int
        writer.add_scalar("averaged_loss/val", averaged_val_loss, epoch_number)
        writer.add_scalar("acc/val", averaged_val_acc, epoch_number)

    return averaged_val_loss, averaged_val_acc
