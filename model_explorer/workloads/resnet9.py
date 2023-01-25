import os
import torch

import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from tqdm import tqdm
from PIL import Image

from model_explorer.utils.logger import logger


device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Cutout(object):
    """
    Implements Cutout regularization as proposed by DeVries and Taylor (2017), https://arxiv.org/pdf/1708.04552.pdf.
    """

    def __init__(self, num_cutouts, size, p=0.5):
        """
        Parameters
        ----------
        num_cutouts : int
            The number of cutouts
        size : int
            The size of the cutout
        p : float (0 <= p <= 1)
            The probability that a cutout is applied (similar to keep_prob for Dropout)
        """
        self.num_cutouts = num_cutouts
        self.size = size
        self.p = p

    def __call__(self, img):

        height, width = img.size

        cutouts = np.ones((height, width))

        if np.random.uniform() < 1 - self.p:
            return img

        for i in range(self.num_cutouts):
            y_center = np.random.randint(0, height)
            x_center = np.random.randint(0, width)

            y1 = np.clip(y_center - self.size // 2, 0, height)
            y2 = np.clip(y_center + self.size // 2, 0, height)
            x1 = np.clip(x_center - self.size // 2, 0, width)
            x2 = np.clip(x_center + self.size // 2, 0, width)

            cutouts[y1:y2, x1:x2] = 0

        cutouts = np.broadcast_to(cutouts, (3, height, width))
        cutouts = np.moveaxis(cutouts, 0, 2)
        img = np.array(img)
        img = img * cutouts
        return Image.fromarray(img.astype('uint8'), 'RGB')


class ResidualBlock(nn.Module):
    """
    A residual block as defined by He et al.
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding,
                 stride):
        super(ResidualBlock, self).__init__()
        self.conv_res1 = nn.Conv2d(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=kernel_size,
                                   padding=padding,
                                   stride=stride,
                                   bias=False)
        self.conv_res1_bn = nn.BatchNorm2d(num_features=out_channels,
                                           momentum=0.9)
        self.conv_res2 = nn.Conv2d(in_channels=out_channels,
                                   out_channels=out_channels,
                                   kernel_size=kernel_size,
                                   padding=padding,
                                   bias=False)
        self.conv_res2_bn = nn.BatchNorm2d(num_features=out_channels,
                                           momentum=0.9)

        if stride != 1:
            # in case stride is not set to 1, we need to downsample the
            # residual so that the dimensions are the same when we add them
            # together
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(num_features=out_channels, momentum=0.9))
        else:
            self.downsample = None

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        residual = x

        out = self.conv_res1(x)
        out = self.conv_res1_bn(out)
        out = self.relu(out)
        out = self.conv_res2(out)
        out = self.conv_res2_bn(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out += residual
        out = self.relu(out)
        return out


class ResNet9(nn.Module):
    """
    A Residual network.
    """

    def __init__(self):
        super(ResNet9, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(num_features=64, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(num_features=128, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(in_channels=128,
                          out_channels=128,
                          kernel_size=3,
                          stride=1,
                          padding=1),
            nn.Conv2d(in_channels=128,
                      out_channels=256,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(num_features=256, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(num_features=256, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(in_channels=256,
                          out_channels=256,
                          kernel_size=3,
                          stride=1,
                          padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc = nn.Linear(in_features=1024, out_features=10, bias=True)

    def forward(self, x):
        out = self.conv(x)
        out = out.view(-1, out.shape[1] * out.shape[2] * out.shape[3])
        out = self.fc(out)
        return out


def save_parameters(model, optimizer, train_accuracies, test_accuracies, path):
    """Saves the parameters of the network to the specified directory.
    """
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_accuracies': train_accuracies,
            'test_accuracies': test_accuracies
        }, path)


def train(model, epochs=10, batch_size=32, state_file="resnet9.pth"):
    model = model.to(device)

    torch.autograd.anomaly_mode.set_detect_anomaly(mode=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = torch.nn.CrossEntropyLoss().cuda()
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    train_accuracies = []
    test_accuracies = []

    train_transform = transforms.Compose([
        Cutout(num_cutouts=2, size=8, p=0.8),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = datasets.CIFAR10('data/cifar',
                                     train=True,
                                     download=True,
                                     transform=train_transform)
    data_loader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size=batch_size,
                                              shuffle=True)

    for epoch_idx in tqdm(range(epochs), desc="Epochs"):
        model.train()

        pbar = tqdm(total=len(data_loader), desc="Curr Epoch")

        epoch_correct = 0
        epoch_total = 0

        for image, target in data_loader:
            image, target = image.to(device), target.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(mode=True):
                output = model(image)
                loss = criterion(output, target.squeeze_())
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(output, dim=1)
                batch_total = target.size(0)
                batch_correct = (predicted == target.flatten()).sum().item()

                epoch_total += batch_total
                epoch_correct += batch_correct

                info_str = 'Last batch accuracy: {:.4f} - Running epoch accuracy {:.4f}'.\
                            format(batch_correct / batch_total, epoch_correct / epoch_total)
                pbar.update(1)
                pbar.desc = info_str

        lr_scheduler.step()
        pbar.close()

        train_accuracies.append(epoch_correct / epoch_total)

        test_acc = test(model, batch_size=batch_size)
        test_accuracies.append(test_acc)
        print("Test accuracy is {:.4f}".format(test_acc))

    save_parameters(model, optimizer, train_accuracies, test_accuracies, state_file)


def test(model, batch_size=32):
    model.eval()

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    test_dataset = datasets.CIFAR10('data/cifar',
                                    train=False,
                                    download=True,
                                    transform=test_transform)

    data_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    correct = 0
    total = 0
    with torch.no_grad():
        for image, target in tqdm(data_loader, desc="Testing"):
            image, target = image.to(device), target.to(device)

            outputs = model(image)
            _, predicted = torch.max(outputs, dim=1)
            total += target.size(0)
            correct += (predicted == target.flatten()).sum().item()

    return correct / total


def resnet9_init():
    model = ResNet9()

    state_file = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                              "./resnet9_state_dict.pt")

    if not os.path.isfile(state_file):
        logger.info("No ResNet9 state dict found at {}, training from scratch".format(state_file))
        train(model, epochs=30, batch_size=128, state_file=state_file)

    checkpoint = torch.load(state_file, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model


model = resnet9_init()
