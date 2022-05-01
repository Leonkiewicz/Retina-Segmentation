import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import imageio
import albumentations as A
from albumentations import HorizontalFlip, VerticalFlip, Rotate

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_data(path):
    X_train = sorted(glob(os.path.join(path, "data", "training", "images", "*.tif")))
    y_train = sorted(glob(os.path.join(path, "data", "training", "1st_manual", "*.gif")))

    X_test = sorted(glob(os.path.join(path, "data", "test", "images", "*.tif")))
    y_test = sorted(glob(os.path.join(path, "data", "test", "1st_manual", "*.gif")))

    return (X_train, y_train), (X_test, y_test)


def augment_data(images, masks, saving_path, resize_w, resize_h, augment=True):
    for idx, (path_x, path_y) in tqdm(enumerate(zip(images, masks)), total=len(images)):
        """ Name extraction """
        fnames_in_path = path_x.split("\\")
        image_name_no_ext = fnames_in_path[-1][:-4]
        # print(" Image name: {}".format(image_name_no_ext))
        # print(path_y)

        """ Reading the image and the mask """
        image = cv2.imread(path_x, cv2.IMREAD_COLOR)
        mask = imageio.mimread(path_y)[0]
        # print(image.shape, mask.shape)


        """ Augmentation         // try mixing augmentations"""
        if augment == True:
            horizontal_flip = HorizontalFlip(p=1.0)
            horizontal_flipped = horizontal_flip(image=image, mask=mask)
            x_hor_flipped = horizontal_flipped["image"]
            y_hor_flipped = horizontal_flipped["mask"]

            vertical_flip = VerticalFlip(p=1.0)
            vertical_flipped = vertical_flip(image=image, mask=mask)
            x_vert_flipped = vertical_flipped["image"]
            y_vert_flipped = vertical_flipped["mask"]

            rotate = Rotate(limit=40, p=1.0)
            rotated = rotate(image=image, mask=mask)
            x_rotated = rotated["image"]
            y_rotated = rotated["mask"]

            transform = A.Compose(
                [
                    A.Rotate(limit=40, p=0.8),
                    A.HorizontalFlip(p=0.7),
                    A.VerticalFlip(p=0.8),
                ]
            )
            transformed = transform(image=image, mask=mask)
            x_transformed = transformed["image"]
            y_transformed = transformed["mask"]

            X = [image, x_hor_flipped, x_vert_flipped, x_rotated, x_transformed]
            Y = [mask, y_hor_flipped, y_vert_flipped, y_rotated, y_transformed]

        else:
            X = [image]
            Y = [mask]

        idx = 0
        for i, m in zip(X, Y):
            i = cv2.resize(i, (resize_w, resize_h))
            m = cv2.resize(m, (resize_w, resize_h))

            temp_img_name = f"{image_name_no_ext}_{idx}.png"
            temp_mask_name = f"{image_name_no_ext}_{idx}.png"

            image_path = os.path.join(saving_path, "images", temp_img_name)
            mask_path = os.path.join(saving_path, "mask", temp_mask_name)

            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)

            idx += 1


if __name__ == "__main__":
    np.random.seed(42)

    data_path = os.getcwd()
    (X_train, y_train), (X_test, y_test) = load_data(data_path)

    print("Amount of training set images: {}".format(len(X_train)))
    print("Amount of training set masks: {}".format(len(y_train)))

    print("Amount of test set images: {}".format(len(X_test)))
    print("Amount of test set masks: {}".format(len(y_test)))


    # Creating directories for the augmented images/masks
    create_dir("augmented_data/training/images/")
    create_dir("augmented_data/training/mask/")
    create_dir("augmented_data/test/images/")
    create_dir("augmented_data/test/mask/")


    augment_data(X_train, y_train, "augmented_data/training/", 512, 512, augment=True)
    augment_data(X_test, y_test, "augmented_data/test/", 512, 512, augment=False)