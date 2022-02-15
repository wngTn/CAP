"""
WiderFace evaluation code
author: wondervictor
mail: tianhengcheng@gmail.com
copyright@wondervictor
"""

import os
import tqdm
import pickle
import argparse
import numpy as np
from scipy.io import loadmat
from bbox import bbox_overlaps
from IPython import embed


def get_gt_boxes(gt_dir):
    computer_node_dict = {'cn01': 0, 'cn02': 1, 'cn03': 2, 'cn04': 3}
    fp_bbox_map = {}
    name = ''
    facebox_list = np.empty((4, 0)).tolist()
    event_list = ['cn01', 'cn02', 'cn03', 'cn04']
    file_list = np.empty((4, 0)).tolist()

    for line in open(gt_dir, 'r'):
        line = line.strip()
        if line.startswith('#'):
            name = line[1:].strip()
            fp_bbox_map[name] = []
            continue

        assert name is not None
        assert name in fp_bbox_map
        fp_bbox_map[name].append(line)

    for key in fp_bbox_map:
        event_node = key.rsplit('/', 1)[0]
        file_name = key.rsplit('/', 1)[1]
        all_boxes = np.zeros((0, 4))
        for aline in fp_bbox_map[key]:
            values = [float(x) for x in aline.strip().split()]
            bbox = np.zeros((1, 4))
            bbox[0][0] = values[0]
            bbox[0][1] = values[1]
            bbox[0][2] = values[2]
            bbox[0][3] = values[3]
            bbox = bbox.astype(int)
            all_boxes = np.append(all_boxes, bbox, axis=0)

        index = computer_node_dict[event_node]
        facebox_list[index].append(all_boxes)
        file_list[index].append(os.path.splitext(file_name)[0])

    cn01_gt_list = [facebox_list[0]]
    cn02_gt_list = [facebox_list[1]]
    cn03_gt_list = [facebox_list[2]]
    cn04_gt_list = [facebox_list[3]]

    return facebox_list, event_list, file_list, cn01_gt_list, cn02_gt_list, cn03_gt_list, cn04_gt_list


def get_gt_boxes_from_txt(gt_path, cache_dir):
    cache_file = os.path.join(cache_dir, 'gt_cache.pkl')
    if os.path.exists(cache_file):
        f = open(cache_file, 'rb')
        boxes = pickle.load(f)
        f.close()
        return boxes

    f = open(gt_path, 'r')
    state = 0
    lines = f.readlines()
    lines = list(map(lambda x: x.rstrip('\r\n'), lines))
    boxes = {}
    print(len(lines))
    f.close()
    current_boxes = []
    current_name = None
    for line in lines:
        if state == 0 and '--' in line:
            state = 1
            current_name = line
            continue
        if state == 1:
            state = 2
            continue

        if state == 2 and '--' in line:
            state = 1
            boxes[current_name] = np.array(current_boxes).astype('float32')
            current_name = line
            current_boxes = []
            continue

        if state == 2:
            box = [float(x) for x in line.split(' ')[:4]]
            current_boxes.append(box)
            continue

    f = open(cache_file, 'wb')
    pickle.dump(boxes, f)
    f.close()
    return boxes


def read_pred_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
        img_file = lines[0].rstrip('\n\r')
        lines = lines[2:]

    # b = lines[0].rstrip('\r\n').split(' ')[:-1]
    # c = float(b)
    # a = map(lambda x: [[float(a[0]), float(a[1]), float(a[2]), float(a[3]), float(a[4])] for a in x.rstrip('\r\n').split(' ')], lines)
    boxes = []
    for line in lines:
        line = line.rstrip('\r\n').split(' ')
        if line[0] == '':
            continue
        # a = float(line[4])
        boxes.append([float(line[0]), float(line[1]), float(line[2]), float(line[3]), float(line[4])])
    boxes = np.array(boxes)
    # boxes = np.array(list(map(lambda x: [float(a) for a in x.rstrip('\r\n').split(' ')], lines))).astype('float')
    return img_file.split('/')[-1], boxes


def get_preds(pred_dir):
    events = os.listdir(pred_dir)
    boxes = dict()
    pbar = tqdm.tqdm(events)

    for event in pbar:
        pbar.set_description('Reading Predictions ')
        event_dir = os.path.join(pred_dir, event)
        event_images = os.listdir(event_dir)
        current_event = dict()
        for imgtxt in event_images:
            imgname, _boxes = read_pred_file(os.path.join(event_dir, imgtxt))
            current_event[imgname.rstrip('.jpg')] = _boxes
        boxes[event] = current_event
    return boxes


def norm_score(pred):
    """ norm score
    pred {key: [[x1,y1,x2,y2,s]]}
    """

    max_score = 0
    min_score = 1

    for _, k in pred.items():
        for _, v in k.items():
            if len(v) == 0:
                continue
            _min = np.min(v[:, -1])
            _max = np.max(v[:, -1])
            max_score = max(_max, max_score)
            min_score = min(_min, min_score)

    diff = max_score - min_score
    for _, k in pred.items():
        for _, v in k.items():
            if len(v) == 0:
                continue
            v[:, -1] = (v[:, -1] - min_score) / diff


def image_eval(pred, gt, ignore, iou_thresh, del_res):
    """ single image evaluation
    pred: Nx5
    gt: Nx4
    ignore:
    """

    _pred = pred.copy()
    _gt = gt.copy()
    pred_recall = np.zeros(_pred.shape[0])
    recall_list = np.zeros(_gt.shape[0])
    proposal_list = np.ones(_pred.shape[0])

    _pred[:, 2] = _pred[:, 2] + _pred[:, 0]
    _pred[:, 3] = _pred[:, 3] + _pred[:, 1]
    _gt[:, 2] = _gt[:, 2] + _gt[:, 0]
    _gt[:, 3] = _gt[:, 3] + _gt[:, 1]

    overlaps = bbox_overlaps(_pred[:, :4], _gt)

    # Deletes the least overlapping boxes if there is more predicted than in the ground truth
    if del_res:
        if len(_pred) > len(_gt):
            for i in range(len(_pred) - len(_gt)):
                smallest_row_index = overlaps.sum(axis=1).argmin()
                overlaps = np.delete(overlaps, smallest_row_index, axis=0)
                pred_recall = pred_recall[:-1]
                proposal_list = proposal_list[:-1]

    #for h in range(_pred.shape[0]):
    for h in range(overlaps.shape[0]):
        gt_overlap = overlaps[h]
        max_overlap, max_idx = gt_overlap.max(), gt_overlap.argmax()
        if max_overlap >= iou_thresh:
            if ignore[max_idx] == 0:
                recall_list[max_idx] = -1
                proposal_list[h] = -1
            elif recall_list[max_idx] == 0:
                recall_list[max_idx] = 1

        r_keep_index = np.where(recall_list == 1)[0]
        pred_recall[h] = len(r_keep_index)
    return pred_recall, proposal_list


def img_pr_info(thresh_num, pred_info, proposal_list, pred_recall):
    # pred_info has to be sorted by threshold (maybe not)
    pr_info = np.zeros((thresh_num, 2)).astype('float')
    for t in range(thresh_num):
        thresh = 1 - (t + 1) / thresh_num
        r_index = np.where(pred_info[:, 4] >= thresh)[0]
        if len(r_index) == 0:
            pr_info[t, 0] = 0
            pr_info[t, 1] = 0
        else:
            r_index = r_index[-1]
            # what we propose how many faces
            p_index = np.where(proposal_list[:r_index + 1] == 1)[0]
            pr_info[t, 0] = len(p_index)
            pr_info[t, 1] = pred_recall[r_index] if r_index < len(pred_recall) else pred_recall[-1]
    return pr_info


def dataset_pr_info(thresh_num, pr_curve, count_face):
    _pr_curve = np.zeros((thresh_num, 2))
    for i in range(thresh_num):
        if pr_curve[i, 0] != 0:
            _pr_curve[i, 0] = pr_curve[i, 1] / pr_curve[i, 0]
        else:
            _pr_curve[i, 0] = 0
        _pr_curve[i, 1] = pr_curve[i, 1] / count_face
    return _pr_curve


def voc_ap(rec, prec):
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def evaluation(pred, gt_path, iou_thresh, del_res):
    pred = get_preds(pred)
    norm_score(pred)
    facebox_list, event_list, file_list, cn01_gt_list, cn02_gt_list, cn03_gt_list, cn04_gt_list = get_gt_boxes(gt_path)
    event_num = len(event_list)
    thresh_num = 1000
    aps = []

    # [hard, medium, easy]
    pbar = tqdm.tqdm(range(event_num))

    for i in pbar:

        count_face = 0
        pr_curve = np.zeros((thresh_num, 2)).astype('float')

        pbar.set_description('Processing {}'.format(event_list[i]))
        event_name = str(event_list[i])
        img_list = file_list[i]
        pred_list = pred[event_name]
        # img_pr_info_list = np.zeros((len(img_list), thresh_num, 2))
        gt_bbx_list = facebox_list[i]

        for j in range(len(img_list)):
            pred_info = pred_list.get(img_list[j])
            if pred_info is None:
                continue

            gt_boxes = gt_bbx_list[j].astype('float')
            # keep_index = sub_gt_list[j]
            count_face += len(gt_boxes)

            if len(gt_boxes) == 0 or len(pred_info) == 0:
                continue

            # sort pred_info by threshold, maybe not necessary
            pred_info = pred_info[np.argsort(pred_info[:, 4])[::-1]]

            ignore = np.ones(gt_boxes.shape[0])
            # if len(keep_index) != 0:
            #     ignore[keep_index - 1] = 1
            pred_recall, proposal_list = image_eval(pred_info, gt_boxes, ignore, iou_thresh, del_res)

            _img_pr_info = img_pr_info(thresh_num, pred_info, proposal_list, pred_recall)

            pr_curve += _img_pr_info
        pr_curve = dataset_pr_info(thresh_num, pr_curve, count_face)

        propose = pr_curve[:, 0]
        recall = pr_curve[:, 1]

        ap = voc_ap(recall, propose)
        aps.append(ap)

    print("==================== Results ====================")
    print("CN01   Val AP: {}".format(aps[0]))
    print("CN02   Val AP: {}".format(aps[1]))
    print("CN03   Val AP: {}".format(aps[2]))
    print("CN04   Val AP: {}".format(aps[3]))
    print("=================================================")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pred', default="./lower_lr_8_combined/ground_truth_1/")
    parser.add_argument('-g', '--gt', default='./ground_truth/labels.txt')
    parser.add_argument('-t', '--threshold', type=float, default=0.4)
    parser.add_argument('-d', '--delete_residual', type=bool, default=True)

    args = parser.parse_args()
    evaluation(args.pred, args.gt, args.threshold, args.delete_residual)
