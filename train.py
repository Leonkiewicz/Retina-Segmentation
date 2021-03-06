import os
import time
from glob import glob
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from xgboost import train

from UNetArchitecture import UNet
from loss import DiceLoss, DiceBCELoss
from utils import seeding, epoch_time
from data_preprocessing import augment_data, create_dir
from data import RetinaDataset


def train_fn(model, loader, optimizer, loss_fn, device):
    loop = tqdm(loader)

    epoch_loss = 0.0
    model.train()

    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(device=device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        loop.set_postfix(loss=loss.item())

    epoch_loss = epoch_loss/len(loader)
    return epoch_loss


def evaluate(model, loader, loss_fn, device):
    epoch_loss = 0.0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()
        
        epoch_loss /= len(loader)
    return epoch_loss




if __name__ == "__main__":
    seeding(42)

    create_dir("files")

    """ Load the dataset """
    cur_dir = os.getcwd()
    X_train = sorted(glob(os.path.join(cur_dir, "augmented_data", "training", "images", "*.png")))
    y_train = sorted(glob(os.path.join(cur_dir, "augmented_data", "training", "mask", "*.png")))

    X_test = sorted(glob(os.path.join(cur_dir, "augmented_data", "test", "images", "*.png")))
    y_test = sorted(glob(os.path.join(cur_dir, "augmented_data", "test", "mask", "*.png")))

    print("Amount of training set images: {}".format(len(X_train)))
    print("Amount of training set masks: {}".format(len(y_train)))

    print("Amount of test set images: {}".format(len(X_test)))
    print("Amount of test set masks: {}".format(len(y_test)))


    """ Hyperparameters """
    H = 512
    W = 512 
    size = (H, W)
    batch_size = 2
    num_epochs = 50
    learning_rate = 1e-4
    checkpoint_path = "files/checkpoint.pth"


    """ Dataset and DataLoader """
    train_dataset = RetinaDataset(X_train, y_train)
    test_dataset = RetinaDataset(X_test, y_test)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet()
    model = model.to(device)

    optimizer = torch.optim.NAdam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    loss_fn = DiceBCELoss()

    best_test_loss = float("inf")

    """ Training loop """
    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss = train_fn(model, train_loader, optimizer, loss_fn, device)
        test_loss = evaluate(model, test_loader, loss_fn, device)

        """ Saving the model """
        if test_loss < best_test_loss:
            print(f"Test loss value: {test_loss:.3f}")
            best_test_loss = test_loss
            torch.save(model.state_dict(), checkpoint_path)


        finish_time = time.time()
        time_elapsed = epoch_time(start_time, finish_time)

        data_str = f'Epoch: {epoch+1:02} | Time elapsed: {time_elapsed:.2f}s\n'
        data_str += f'\tTrain loss: {train_loss:.3f}\n'
        data_str += f'\t Test loss: {test_loss:.3f}\n'
        print(data_str)

