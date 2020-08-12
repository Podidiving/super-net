from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms


def get_train_dataloader(
    train_root: str, batch_size: int = 256, num_workers: int = 10
):
    train_transforms = transforms.Compose([transforms.ToTensor()])
    dataset = MNIST(
        train_root, train=True, transform=train_transforms, download=True
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    return dataloader


def get_test_dataloader(
    test_root: str, batch_size: int = 256, num_workers: int = 10
):
    test_transforms = transforms.Compose([transforms.ToTensor()])
    dataset = MNIST(
        test_root, train=False, transform=test_transforms, download=True
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return dataloader
