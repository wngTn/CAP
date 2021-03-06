import cv2
import numpy as np
import random
import albumentations as A
from utils.box_utils import matrix_iof


def _rotate(image, boxes, landm, labels):
    for row in boxes:
        row[2] -= row[0]
        row[3] -= row[1]

    boxes = np.insert(boxes, 4, -1, axis=1)

    landm_temp = []
    i = 0
    for row in landm:
        for j in range(0, len(row), 2):
            if row[j] > 0 and row[j + 1] > 0:
                if abs(row[j]-image.shape[1]) < 0.0000001:
                    row[j] -= 0.0000001
                if abs(row[j+1]-image.shape[0]) < 0.0000001:
                    row[j+1] -= 0.0000001
                landm_temp += [(row[j], row[j + 1], i)]
                i += 1

    transform = A.Compose(
        [
            A.Rotate(border_mode=cv2.BORDER_CONSTANT, p=1)
        ], bbox_params=A.BboxParams(format='coco', min_area=512, min_visibility=0.5)
        , keypoint_params=A.KeypointParams(format='xy')
    )

    augmentations = transform(image=image, bboxes=boxes, keypoints=landm_temp)
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

    landm = np.zeros((0, 10))
    n = bbox.shape[0]
    for i in range(n):
        temp = np.zeros((1, 10))
        temp.fill(-1)
        landm = np.append(landm, temp, axis=0)

    for (x, y, z) in augmented_landm:
        person = z // 2
        if person >= n:
            continue
        # 1 left, 0 right
        eye = z % 2
        if eye == 1:
            landm[person][2] = x
            landm[person][3] = y
        else:
            landm[person][0] = x
            landm[person][1] = y

    # TODO wouldn't work if anything other than 1 as label
    if n - len(labels) < 0:
        labels = labels[:n-len(labels)]

    return augmented_img, bbox, landm, labels


def _albumentation(image):
    transform = A.Compose(
        [
            A.MotionBlur(blur_limit=(3, 5), p=0.15),
            # A.CoarseDropout(max_holes=10, max_height=10, max_width=10, fill_value=0, p=0.2),
            # A.Blur(blur_limit=(3, 10), p=0.3),
        ]
    )
    augmentations = transform(image=image)
    return augmentations["image"]


def _crop(image, boxes, labels, landm, img_dim):
    height, width, _ = image.shape
    pad_image_flag = True

    for _ in range(250):
        """
        if random.uniform(0, 1) <= 0.2:
            scale = 1.0
        else:
            scale = random.uniform(0.3, 1.0)
        """
        PRE_SCALES = [0.3, 0.45, 0.6, 0.8, 1.0]
        scale = random.choice(PRE_SCALES)
        short_side = min(width, height)
        w = int(scale * short_side)
        h = w

        if width == w:
            l = 0
        else:
            l = random.randrange(width - w)
        if height == h:
            t = 0
        else:
            t = random.randrange(height - h)
        roi = np.array((l, t, l + w, t + h))

        value = matrix_iof(boxes, roi[np.newaxis])
        flag = (value >= 1)
        if not flag.any():
            continue

        centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        mask_a = np.logical_and(roi[:2] < centers, centers < roi[2:]).all(axis=1)
        boxes_t = boxes[mask_a].copy()
        labels_t = labels[mask_a].copy()
        landms_t = landm[mask_a].copy()
        landms_t = landms_t.reshape([-1, 5, 2])

        if boxes_t.shape[0] == 0:
            continue

        image_t = image[roi[1]:roi[3], roi[0]:roi[2]]

        boxes_t[:, :2] = np.maximum(boxes_t[:, :2], roi[:2])
        boxes_t[:, :2] -= roi[:2]
        boxes_t[:, 2:] = np.minimum(boxes_t[:, 2:], roi[2:])
        boxes_t[:, 2:] -= roi[:2]

        # landm
        landms_t[:, :, :2] = landms_t[:, :, :2] - roi[:2]
        landms_t[:, :, :2] = np.maximum(landms_t[:, :, :2], np.array([0, 0]))
        landms_t[:, :, :2] = np.minimum(landms_t[:, :, :2], roi[2:] - roi[:2])
        landms_t = landms_t.reshape([-1, 10])

        # make sure that the cropped image contains at least one face > 16 pixel at training image scale
        b_w_t = (boxes_t[:, 2] - boxes_t[:, 0] + 1) / w * img_dim
        b_h_t = (boxes_t[:, 3] - boxes_t[:, 1] + 1) / h * img_dim
        mask_b = np.minimum(b_w_t, b_h_t) > 0.0
        boxes_t = boxes_t[mask_b]
        labels_t = labels_t[mask_b]
        landms_t = landms_t[mask_b]

        if boxes_t.shape[0] == 0:
            continue

        pad_image_flag = False

        return image_t, boxes_t, labels_t, landms_t, pad_image_flag
    return image, boxes, labels, landm, pad_image_flag


def _distort(image):
    def _convert(image, alpha=1, beta=0):
        tmp = image.astype(float) * alpha + beta
        tmp[tmp < 0] = 0
        tmp[tmp > 255] = 255
        image[:] = tmp

    image = image.copy()

    if random.randrange(2):

        # brightness distortion
        if random.randrange(2):
            _convert(image, beta=random.uniform(-32, 32))

        # contrast distortion
        if random.randrange(2):
            _convert(image, alpha=random.uniform(0.5, 1.5))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # saturation distortion
        if random.randrange(2):
            _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

        # hue distortion
        if random.randrange(2):
            tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
            tmp %= 180
            image[:, :, 0] = tmp

        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    else:

        # brightness distortion
        if random.randrange(2):
            _convert(image, beta=random.uniform(-32, 32))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # saturation distortion
        if random.randrange(2):
            _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

        # hue distortion
        if random.randrange(2):
            tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
            tmp %= 180
            image[:, :, 0] = tmp

        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

        # contrast distortion
        if random.randrange(2):
            _convert(image, alpha=random.uniform(0.5, 1.5))

    return image


def _expand(image, boxes, fill, p):
    if random.randrange(2):
        return image, boxes

    height, width, depth = image.shape

    scale = random.uniform(1, p)
    w = int(scale * width)
    h = int(scale * height)

    left = random.randint(0, w - width)
    top = random.randint(0, h - height)

    boxes_t = boxes.copy()
    boxes_t[:, :2] += (left, top)
    boxes_t[:, 2:] += (left, top)
    expand_image = np.empty(
        (h, w, depth),
        dtype=image.dtype)
    expand_image[:, :] = fill
    expand_image[top:top + height, left:left + width] = image
    image = expand_image

    return image, boxes_t


def _mirror(image, boxes, landms):
    _, width, _ = image.shape
    if random.randrange(2):
        image = image[:, ::-1]
        boxes = boxes.copy()
        boxes[:, 0::2] = width - boxes[:, 2::-2]

        # landm
        landms = landms.copy()
        landms = landms.reshape([-1, 5, 2])
        landms[:, :, 0] = width - landms[:, :, 0]
        tmp = landms[:, 1, :].copy()
        landms[:, 1, :] = landms[:, 0, :]
        landms[:, 0, :] = tmp
        tmp1 = landms[:, 4, :].copy()
        landms[:, 4, :] = landms[:, 3, :]
        landms[:, 3, :] = tmp1
        landms = landms.reshape([-1, 10])

    return image, boxes, landms


def _pad_to_square(image, rgb_mean, pad_image_flag):
    if not pad_image_flag:
        return image
    height, width, _ = image.shape
    long_side = max(width, height)
    image_t = np.empty((long_side, long_side, 3), dtype=image.dtype)
    image_t[:, :] = rgb_mean
    image_t[0:0 + height, 0:0 + width] = image
    return image_t


def _resize_subtract_mean(image, insize, rgb_mean):
    interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
    interp_method = interp_methods[random.randrange(5)]
    image = cv2.resize(image, (insize, insize), interpolation=interp_method)
    image = image.astype(np.float32)
    image -= rgb_mean
    return image.transpose(2, 0, 1)


class preproc(object):

    def __init__(self, img_dim, rgb_means):
        self.img_dim = img_dim
        self.rgb_means = rgb_means

    def __call__(self, image, targets):
        assert targets.shape[0] > 0, "this image does not have gt"

        boxes = targets[:, :4].copy()
        labels = targets[:, -1].copy()
        landm = targets[:, 4:-1].copy()

        image_t, boxes_t, labels_t, landm_t, pad_image_flag = _crop(image, boxes, labels, landm, self.img_dim)
        image_t = _distort(image_t)
        image_t = _albumentation(image_t)
        image_t = _pad_to_square(image_t, self.rgb_means, pad_image_flag)
        image_t, boxes_t, landm_t = _mirror(image_t, boxes_t, landm_t)
        # image_t, boxes_t, landm_t, labels_t = _rotate(image_t, boxes_t, landm_t, labels_t)
        height, width, _ = image_t.shape
        image_t = _resize_subtract_mean(image_t, self.img_dim, self.rgb_means)

        boxes_t[:, 0::2] = boxes_t[:, 0::2]/width * self.img_dim
        boxes_t[:, 1::2] = boxes_t[:, 1::2]/height * self.img_dim
        landm_t[:, 0::2] = landm_t[:, 0::2]/width * self.img_dim
        landm_t[:, 1::2] = landm_t[:, 1::2]/height * self.img_dim

        # boxes_t[:, 0::2] /= width
        # boxes_t[:, 1::2] /= height
        #
        # landm_t[:, 0::2] /= width
        # landm_t[:, 1::2] /= height

        labels_t = np.expand_dims(labels_t, 1)
        targets_t = np.hstack((boxes_t, landm_t, labels_t))

        return image_t, targets_t
