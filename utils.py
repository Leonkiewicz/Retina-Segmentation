import os
import time
import random
import numpy as np
import cv2
import torch


def seeding(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def epoch_time(start_time, finish_time):
    """ Calculate the time it took to train the model"""
    elapsed_time = finish_time - start_time
    return elapsed_time

