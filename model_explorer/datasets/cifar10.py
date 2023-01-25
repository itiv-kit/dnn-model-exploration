import torchvision.transforms as transforms
import torchvision.datasets as datasets


def prepare_cifar10(**kwargs):
    transfs = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    dataset = datasets.CIFAR10('data/cifar',
                               train=False,
                               download=True,
                               transform=transfs)

    return dataset


dataset_creator = prepare_cifar10
collate_fn = None
