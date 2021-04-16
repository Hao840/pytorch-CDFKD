from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST


def get_loader(root, dataset, batch_size):
    if dataset == 'MNIST':
        transform_train = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        transform_test = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        data_train = MNIST(root=root,
                           download=True,
                           transform=transform_train)
        data_test = MNIST(root=root,
                          train=False,
                          download=True,
                          transform=transform_test)

    elif dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        data_train = CIFAR10(root=root,
                             download=True,
                             transform=transform_train)
        data_test = CIFAR10(root=root,
                            train=False,
                            download=True,
                            transform=transform_test)
    else:
        raise NotImplementedError

    data_train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
    data_test_loader = DataLoader(data_test, batch_size=batch_size)
    return data_train_loader, data_test_loader
