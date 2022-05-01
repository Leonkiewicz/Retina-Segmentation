# Retina-Segmentation

The code in this repository implements the UNet Architecture in PyTorch to perform the semantic segmentation on the Digital Retinal Images for Vessel Extraction dataset.
![unet](https://user-images.githubusercontent.com/64259364/166144011-fac5080a-8e02-4c35-920b-0f549476de99.png)


# Dataset
The original dataset contains 40 images and 40 masks (20 for training and 20 for evaluation). Using data augmentation the training dataset was increased to 50 images and 50 masks. 

# Results
During the evaluation trained model achieved the f1 score of 0.794 and Dice score of 0.777.
Image below shows the original image, ground-truth mask and mask created by the trained model:
![18_test_0result](https://user-images.githubusercontent.com/64259364/166137913-8a87bdf4-edbe-487c-b0a1-626dbb791291.png)
