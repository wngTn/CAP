import cv2
import numpy as np
import albumentations as A
import os
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
import torch.utils.data as data
from data import WiderFaceDetection, detection_collate, preproc, data_augment
from layers.modules import MultiBoxLoss
from layers.functions.prior_box import PriorBox
import time
import datetime
import math
from models.retinaface import RetinaFace


def customAug():
    prepoc = preproc(840, (79, 92, 98))

    labels = "/home/tony/Documents/CAP/src/data/widerface/train/label1.txt"
    image_pre = "/home/tony/Documents/CAP/src/data/widerface/train/images/"

    dataset = WiderFaceDetection(labels, prepoc)

    img, targets = dataset[102]

    img = np.array(img)

    img = img.transpose(2, 0, 1)
    img = img.transpose(2, 0, 1)

    targets = targets.astype(int)

    cv2.rectangle(img, (targets[0, 0], targets[0, 1]), (targets[0, 2], targets[0, 3]), (0, 0, 255), 2)
    cv2.circle(img, (targets[0, 4], targets[0, 5]), 1, (0, 0, 255), 2)
    cv2.circle(img, (targets[0, 6], targets[0, 7]), 1, (0, 0, 255), 2)

    cv2.imwrite('WHAAT.jpg', img)


def autAug():
    img = cv2.imread("../media/girl.jpg")
    transform = A.Compose(
        [
            A.MotionBlur(blur_limit=(10, 25), p=0.5),
            A.Cutout(num_holes=10, max_h_size=10, max_w_size=10, fill_value=0, p=1),
            A.Blur(blur_limit=(5, 10), p=0.3),
        ]
    )
    augmentations = transform(image=img)
    augmented_img = augmentations["image"]

    cv2.imwrite('WHAAT.jpg', augmented_img)





if __name__ == '__main__':
    autAug()
