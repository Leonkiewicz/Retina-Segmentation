import os
from operator import add 
import time
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import imageio
import torch

from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score

from UNetArchitecture import UNet
from utils import epoch_time, seeding
from data_preprocessing import create_dir



def calculate_metrics(y_true, y_pred):
    y_true = y_true.cpu().numpy()
    y_true = y_true > 0.5
    y_true = y_true.astype(np.uint8)
    y_true = y_true.reshape(-1)

    y_pred = y_pred.cpu().numpy()
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.uint8)
    y_pred = y_pred.reshape(-1)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    jaccard = jaccard_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    return [acc, f1, jaccard, precision, recall]


def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)
    mask = np.concatenate([mask, mask, mask], axis=-1)
    return mask


if __name__ == "__main__":
    seeding(42)

    create_dir("results")

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
    num_epochs = 10
    learning_rate = 1e-4
    checkpoint_path = "files/checkpoint.pth"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet()
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    num_correct = 0
    num_pixels = 0
    dice_score = 0
    metrics_score = [0., 0., 0., 0., 0.]
    time_elapsed = []

    for i, (path_x, path_y) in tqdm(enumerate(zip(X_test, y_test)), total=len(X_test)):
        """ Name extraction """
        fnames_in_path = path_x.split("\\")
        image_name_no_ext = fnames_in_path[-1][:-4]

        """ Reading the image and the mask """
        image = cv2.imread(path_x, cv2.IMREAD_COLOR)
        path_x = np.transpose(image, (2, 0, 1))
        path_x = path_x/255.0
        path_x = np.expand_dims(path_x, axis=0)
        path_x = path_x.astype(np.float32)
        path_x = torch.from_numpy(path_x)
        path_x = path_x.to(device)

        mask = cv2.imread(path_y, cv2.IMREAD_GRAYSCALE)
        path_y = np.expand_dims(mask, axis=0)
        path_y = path_y/255.0
        path_y = np.expand_dims(path_y, axis=0)
        path_y = path_y.astype(np.float32)
        path_y = torch.from_numpy(path_y)
        path_y = path_y.to(device).unsqueeze(1)


        with torch.no_grad():
            start_time = time.time()
            y_pred = model(path_x)
            y_pred = torch.sigmoid(y_pred)
            finish_time = time.time()
            time_elapsed.append(epoch_time(start_time, finish_time))

            score = calculate_metrics(path_y, y_pred) 
            metrics_score = list(map(add, metrics_score, score))
            y_pred = y_pred[0].cpu().numpy()
            y_pred = np.squeeze(y_pred, axis=0)
            y_pred = y_pred > 0.5
            y_pred = np.array(y_pred, dtype=np.uint8) 

            """ Dice score """
            preds = torch.sigmoid(model(path_x))
            preds = (preds > 0.5).float()
            num_correct += (preds == path_y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * path_y).sum()) / (
                (preds + path_y).sum() + 1e-8
            )


        """ Saving the masks """
        original_mask = mask_parse(mask)
        y_pred = mask_parse(y_pred)

        concat_imgs = np.concatenate(
            [image, original_mask, y_pred * 255], axis=1
        )

        save_path = os.path.join(os.getcwd(), "results", f"{image_name_no_ext}result.png")
        cv2.imwrite(save_path, concat_imgs)


    acc = metrics_score[0]/len(X_test)
    f1 = metrics_score[1]/len(X_test)
    jaccard = metrics_score[2]/len(X_test)
    precision = metrics_score[3]/len(X_test)
    recall = metrics_score[4]/len(X_test)
    
    print("Accuracy score: {:.3f}\nF1 score: {:.3f}\nJaccard score: {:.3f}\nPrecision score: {:.3f}\nRecall score: {:.3f}".format(
        acc, f1, jaccard, precision, recall
    ))

    print("Dice score: {:.3f}".format(dice_score/len(X_test)))
