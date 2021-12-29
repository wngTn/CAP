import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

# files = os.listdir("/home/tony/Documents/CAP/src/data/widerface/train/images")
# files.sort()

if __name__ == '__main__':
    files = {"mafa/train_00001990.jpg"}

    labels = "/home/tony/Documents/CAP/src/data/widerface/train/label.txt"
    image_pre = "/home/tony/Documents/CAP/src/data/widerface/train/images/"

    fp_bbox_map = {}

    for line in open(labels, 'r'):
        line = line.strip()
        if line.startswith('#'):
            name = line[1:].strip()
            fp_bbox_map[name] = []
            continue
        assert name is not None
        assert name in fp_bbox_map
        fp_bbox_map[name].append(line)

    for filename in files:
        img = cv2.imread(image_pre + filename)
        for aline in fp_bbox_map[filename]:
            values = [float(x) for x in aline.strip().split()]
            bbox = np.array([values[0], values[1], values[0] + values[2], values[1] + values[3]])
            box = bbox.astype(int)
            kkps = np.array([values[4], values[5], values[7], values[8]])
            kps = kkps.astype(int)
            color = (0, 0, 255)
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.circle(img, (kps[0], kps[1]), 1, color, 2)
            cv2.circle(img, (kps[2], kps[3]), 1, color, 2)

        cv2.imwrite('test.jpg', img)
        # plt.figure()
        # plt.imshow(img[:, :, ::-1])
        # plt.show()
