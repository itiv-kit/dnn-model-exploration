# the code to setup LeNet has been copied (with small alternations) from:
# https://github.com/erykml/medium_articles/blob/master/Computer%20Vision/lenet5_pytorch.ipynb

import numpy as np
from datetime import datetime
import os
from progress.bar import Bar

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import mnist

# check device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# parameters
RANDOM_SEED = 42
LEARNING_RATE = 0.001
N_EPOCHS = 15

IMG_SIZE = 32
N_CLASSES = 10


class LeNet5(nn.Module):

    def __init__(self, n_classes):
        super(LeNet5, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120,
                      kernel_size=5, stride=1),
            nn.Tanh()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=n_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=1)
        return probs


def get_accuracy(model, data_loader, device, verbose=False):
    '''
    Function for computing the accuracy of the predictions over the entire data_loader
    '''

    dataset_size = dataset_size = len(data_loader.dataset)
    if verbose:
        bar = Bar('Processing', max=dataset_size)

    dev = torch.device(device)
    model.to(dev)

    correct_pred = 0
    n = 0

    with torch.no_grad():
        model.eval()
        for X, y_true in data_loader:

            X = X.to(dev)
            y_true = y_true.to(dev)

            _, y_prob = model(X)
            _, predicted_labels = torch.max(y_prob, 1)

            n += y_true.size(0)
            correct_pred += (predicted_labels == y_true).sum()

            if verbose:
                bar.goto(n)

    if verbose:
        bar.finish()

    return correct_pred.float() / n


def train(train_loader, model, criterion, optimizer, device):
    '''
    Function for the training step of the training loop
    '''

    model.train()
    running_loss = 0

    for X, y_true in train_loader:

        optimizer.zero_grad()

        X = X.to(device)
        y_true = y_true.to(device)

        # Forward pass
        y_hat, _ = model(X)
        loss = criterion(y_hat, y_true)
        running_loss += loss.item() * X.size(0)

        # Backward pass
        loss.backward()
        optimizer.step()

    epoch_loss = running_loss / len(train_loader.dataset)
    return model, optimizer, epoch_loss


def validate(valid_loader, model, criterion, device):
    '''
    Function for the validation step of the training loop
    '''

    model.eval()
    running_loss = 0

    for X, y_true in valid_loader:

        X = X.to(device)
        y_true = y_true.to(device)

        # Forward pass and record loss
        y_hat, _ = model(X)
        loss = criterion(y_hat, y_true)
        running_loss += loss.item() * X.size(0)

    epoch_loss = running_loss / len(valid_loader.dataset)

    return model, epoch_loss


def training_loop(model, criterion, optimizer, train_loader, valid_loader, epochs, device, print_every=1):
    '''
    Function defining the entire training loop
    '''

    # set objects for storing metrics
    best_loss = 1e10
    train_losses = []
    valid_losses = []

    # Train model
    for epoch in range(0, epochs):

        # training
        model, optimizer, train_loss = train(
            train_loader, model, criterion, optimizer, device)
        train_losses.append(train_loss)

        # validation
        with torch.no_grad():
            model, valid_loss = validate(
                valid_loader, model, criterion, device)
            valid_losses.append(valid_loss)

        if epoch % print_every == (print_every - 1):

            train_acc = get_accuracy(model, train_loader, device=device)
            valid_acc = get_accuracy(model, valid_loader, device=device)

            print(f'{datetime.now().time().replace(microsecond=0)} --- '
                  f'Epoch: {epoch}\t'
                  f'Train loss: {train_loss:.4f}\t'
                  f'Valid loss: {valid_loss:.4f}\t'
                  f'Train accuracy: {100 * train_acc:.2f}\t'
                  f'Valid accuracy: {100 * valid_acc:.2f}')

    return model, optimizer, (train_losses, valid_losses)

def init_model():

    transform = mnist.get_data_transform()

    train_loader = mnist.get_train_loader(transform)

    valid_loader = mnist.get_validation_loader(transform)

    torch.manual_seed(RANDOM_SEED)

    model = LeNet5(N_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    model, optimizer, _ = training_loop(
        model, criterion, optimizer, train_loader, valid_loader, N_EPOCHS, DEVICE)

    torch.save(model.state_dict(), os.path.join(
        os.path.dirname(__file__), "lenet5_state_dict.pt"))


def leNet5_init():

    model = LeNet5(N_CLASSES)

    state_file = os.path.join(os.path.abspath(
        os.path.dirname(__file__)), "./lenet5_state_dict.pt")

    if not os.path.isfile(state_file):
        print("No model found at {}, training from scratch".format(state_file))
        init_model()

    assert os.path.isfile(
        state_file), "Failed to create and load lenet5_state_dict.pt for pretrained LeNet5."
    model.load_state_dict(torch.load(state_file))

    return model


model = leNet5_init()


if __name__ == "main":
    init_model()
