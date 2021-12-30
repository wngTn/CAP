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
    prepoc = preproc(840, (-20, -28, -32))

    labels = "/home/tony/Documents/CAP/src/data/widerface/train/label1.txt"
    image_pre = "/home/tony/Documents/CAP/src/data/widerface/train/images/"

    dataset = WiderFaceDetection(labels, prepoc)

    img, targets = dataset[28]

    img = np.array(img)

    img = img.transpose(2, 0, 1)
    img = img.transpose(2, 0, 1)

    targets = targets.astype(int)

    for t in targets:
        cv2.rectangle(img, (t[0], t[1]), (t[2], t[3]), (0, 0, 255), 2)
        cv2.circle(img, (t[4], t[5]), 1, (0, 0, 255), 2)
        cv2.circle(img, (t[6], t[7]), 1, (0, 0, 255), 2)

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


# 199, 51, 327, 250

def anno():
    img = cv2.imread("data/widerface/train/images/mafa/train_00000088.jpg")

    bbox = np.zeros((3, 4))
    bbox[0] = [18, 23, 93 + 18, 93 + 23]
    bbox[1] = [89, 87, 97+89, 97+87]
    bbox[2] = [187, 79, 66+187, 66+79]

    landm = np.zeros((3, 10))
    landm[0] = [35.0, 44.0, 75.0, 51.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
    landm[1] = [114.0, 108.0, 159.0, 109.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
    landm[2] = [201.0, 92.0, 232.0, 86.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]

    for row in bbox:
        row[2] -= row[0]
        row[3] -= row[1]

    bbox = np.insert(bbox, 4, -1, axis=1)

    # landm_temp = np.zeros((0, 2))
    # for row in landm:
    #     np.append(landm_temp, [(row[0], row[1]), (row[2], row[3]), (row[4], row[5]), (row[6], row[7]), (row[8], row[9])], axis=0)

    landm_temp = []
    i = 0
    for row in landm:
        for j in range(0, len(row), 2):
            if row[j] > 0 and row[j + 1] > 0:
                landm_temp += [(row[j], row[j + 1], i)]
                i += 1


    #     print([(row[0], row[1]), (row[2], row[3]), (row[4], row[5]), (row[6], row[7]), (row[8], row[9])])
    #     landm_temp += [(row[0], row[1]), (row[2], row[3]), (row[4], row[5]), (row[6], row[7]), (row[8], row[9])]


    transform = A.Compose(
        [
            A.MotionBlur(blur_limit=(10, 25), p=0.5),
            A.Blur(blur_limit=(5, 10), p=0.3),
            A.ShiftScaleRotate(p=1)
        ], bbox_params=A.BboxParams(format='coco', min_area=1024, min_visibility=0.5)
        , keypoint_params=A.KeypointParams(format='xy')
    )

    augmentations = transform(image=img, bboxes=bbox, keypoints=landm_temp)
    augmented_img = augmentations["image"]
    augmented_bbox = augmentations["bboxes"]
    augmented_landm = augmentations["keypoints"]

    bbox = np.zeros((0, 4))
    for b in augmented_bbox:
        bnp = np.zeros((1, 4))
        bnp[0, 0] = b[0]
        bnp[0, 1] = b[1]
        bnp[0, 2] = b[2] + b[0]
        bnp[0, 3] = b[3] + b[1]
        bbox = np.append(bbox, bnp, axis=0)
        b = list(map(int, b))
        cv2.rectangle(augmented_img, (b[0], b[1]), (b[0]+b[2], b[1]+b[3]), (0, 0, 255), 2)
    print(bbox)

    print(augmented_landm)
    landm = np.zeros((0, 10))
    (_, _, l_z) = augmented_landm[-1]
    print(l_z//2)
    for i in range((l_z//2)+1):
        temp = np.zeros((1, 10))
        temp.fill(-1)
        landm = np.append(landm, temp, axis=0)

    print(landm)

    for (x, y, z) in augmented_landm:
        person = z // 2
        # 1 left, 0 right
        eye = z % 2
        if eye == 1:
            landm[person][2] = x
            landm[person][3] = y
        else:
            landm[person][0] = x
            landm[person][1] = y

    print(augmented_landm)
    print(landm)



    for row in landm:
        row = list(map(int, row))
        cv2.circle(augmented_img, (row[0], row[1]), 1, (0, 0, 255), 4)
        cv2.circle(augmented_img, (row[2], row[3]), 1, (0, 255, 255), 4)
        cv2.circle(augmented_img, (row[4], row[5]), 1, (255, 0, 255), 4)
        cv2.circle(augmented_img, (row[6], row[7]), 1, (0, 255, 0), 4)
        cv2.circle(augmented_img, (row[8], row[9]), 1, (255, 0, 0), 4)

    cv2.imwrite('WHAAT.jpg', augmented_img)






    # cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
    # cx = b[0]
    # cy = b[1] + 12
    # cv2.putText(img_raw, text, (cx, cy),
    #             cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
    #
    # # landms
    # cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
    # cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
    # cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
    # cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
    # cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)


if __name__ == '__main__':
    customAug()
