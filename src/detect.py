from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import print_function

import glob
import shutil
from models.retinaface import RetinaFace
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision

import _init_paths
import models

from config import cfg
from config import update_config
from core.inference import get_multi_stage_outputs
from core.inference import aggregate_results
from core.nms import pose_nms
from core.match import match_pose_to_heatmap
from utils.transforms import resize_align_multi_scale
from utils.transforms import get_final_preds
from utils.transforms import get_multi_scale_size
from utils.transforms import up_interpolate
import os

import argparse
import torch
import numpy as np
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2

from utils.box_utils import decode, decode_landm
import time

from statistics import mean

parser = argparse.ArgumentParser(description='Retinaface')

parser.add_argument('-m', '--trained_model', default='./weights/fifth_try/Resnet50_iteration_4000.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('--vis_thres', default=0.42, type=float, help='visualization_threshold')
parser.add_argument('--vis_thres_pose', default=0.3, type=float, help='visualization_threshold for key pose estimation')
parser.add_argument('--cfg', default='../experiments/coco/w48/w48_4x_reg03_bs5_640_adam_lr1e-3_coco_x140.yaml',
                    type=str)
parser.add_argument('--outputDir', type=str, default='output/')
parser.add_argument('--write_scores', type=bool, default=False, help='whether the scores should be written')
parser.add_argument('-d', '--detect', type=str, default='combined', help='What should be detected')
parser.add_argument('--text', type=bool, default=False, help='whether the results should be written on a text file')
parser.add_argument('--save', type=bool, default=True, help='indicates whether the pictures should be saved')
parser.add_argument('opts',
                    help='Modify config options using the command-line',
                    default=None,
                    nargs=argparse.REMAINDER)

args = parser.parse_args()

CTX = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def get_pose_estimation_prediction(cfg, model, image, vis_thre, transforms):
    # size at scale 1.0
    base_size, center, scale = get_multi_scale_size(
        image, cfg.DATASET.INPUT_SIZE, 1.0, 1.0
    )

    with torch.no_grad():
        heatmap_sum = 0
        poses = []

        for scale in sorted(cfg.TEST.SCALE_FACTOR, reverse=True):
            image_resized, center, scale_resized = resize_align_multi_scale(
                image, cfg.DATASET.INPUT_SIZE, scale, 1.0
            )

            image_resized = transforms(image_resized)
            image_resized = image_resized.unsqueeze(0).cuda()

            heatmap, posemap = get_multi_stage_outputs(
                cfg, model, image_resized, cfg.TEST.FLIP_TEST
            )
            heatmap_sum, poses = aggregate_results(
                cfg, heatmap_sum, poses, heatmap, posemap, scale
            )

        heatmap_avg = heatmap_sum / len(cfg.TEST.SCALE_FACTOR)
        poses, scores = pose_nms(cfg, heatmap_avg, poses)

        if len(scores) == 0:
            return [], []
        else:
            if cfg.TEST.MATCH_HMP:
                poses = match_pose_to_heatmap(cfg, poses, heatmap_avg)

            final_poses = get_final_preds(
                poses, center, scale_resized, base_size
            )

        final_results = []
        final_scores = []
        for i in range(len(scores)):
            if scores[i] > vis_thre:
                final_results.append(final_poses[i])
                final_scores.append(scores[i])

        if len(final_results) == 0:
            return [], []

    return final_results, final_scores


def pose(img_raw):
    # transformation
    pose_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    update_config(cfg, args)

    pose_model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(
        cfg, is_train=False
    )

    # if cfg.TEST.MODEL_FILE:
    # cfg.TEST.MODEL_FILE = '../model/pose_coco/pose_dekr_hrnetw48_coco.pth'
    print('=> loading model from {}'.format('../model/pose_coco/pose_dekr_hrnetw48_coco.pth'))
    pose_model.load_state_dict(torch.load('../model/pose_coco/pose_dekr_hrnetw48_coco.pth'), strict=False)
    # else:
    # raise ValueError('expected model defined in config at TEST.MODEL_FILE')

    pose_model.to(CTX)
    pose_model.eval()

    image_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
    image_pose = image_rgb.copy()
    pose_preds, pose_scores = get_pose_estimation_prediction(cfg, pose_model, image_pose, args.vis_thres_pose,
                                                             transforms=pose_transform)

    return pose_preds, pose_scores


# RETINAFACE #

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def bounding_box(img_raw):
    torch.set_grad_enabled(False)
    cfg = cfg_re50
    # net and model
    net = RetinaFace(cfg=cfg, phase='test')
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)

    # testing begin

    img = np.float32(img_raw)

    im_height, im_width, _ = img.shape
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)

    loc, conf, landms = net(img)  # forward pass

    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

    # ignore low scores
    inds = np.where(scores > args.confidence_threshold)[0]
    boxes = boxes[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:args.top_k]
    boxes = boxes[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, args.nms_threshold)
    # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
    dets = dets[keep, :]
    return dets


def is_in_boxes(boxes, x_avg, y_avg, score):
    """
    Args:
        boxes: the detected boxes
        x_avg: average of the x values of the key poses
        y_avg: average of the y values of the key poses
        score: the score of the key pose detection

    Returns: the boxes with the adapted score. The score will be added with 1.5 times of the score of the
    key points if their average is in a box

    """
    buffer = 5
    for j in range(len(boxes)):
        box = boxes[j]
        [x_least, y_least, x_max, y_max, _] = box
        if ((x_least - buffer) <= x_avg <= (x_max + buffer)) and ((y_least - buffer) <= y_avg <= (y_max + buffer)):
            boxes[j][4] += (score * 1.5)
            if boxes[j][4] > 1:
                boxes[j][4] = 1
            return boxes, True

    return boxes, False


def get_bounding_boxes(poses, pose_score):
    """

    Args:
        poses: the key poses that have been detected
        pose_score: the score of the pose

    Returns: a list of the bounding boxes created from the key poses

    """
    result = []
    for j in range(len(poses)):
        coords = poses[j]
        x_values = [tup[0] for tup in coords]
        y_values = [tup[1] for tup in coords]
        x_least = min(x_values) - 7
        x_max = max(x_values) + 7
        y_least = min(y_values) - 10
        y_max = max(y_values) + 15
        result.append([x_least, y_least, x_max, y_max, pose_score[j]])
    return result


def final_bounding_boxes(boxes, poses, pose_score):
    """

    Args:
        boxes: the detected bounding boxes
        poses: the detected key poses
        pose_score: the sores of the detected key poses

    Returns: the final bounding boxes with the scores

    """
    new_poses = []
    new_scores = []
    for j in range(len(poses)):
        coords = poses[j]
        x_avg = mean([tup[0] for tup in coords])
        y_avg = mean([tup[1] for tup in coords])
        boxes_new_scores, in_boxes = is_in_boxes(boxes, x_avg, y_avg, pose_score[j])
        if not in_boxes:
            new_poses.append(coords)
            new_scores.append(pose_score[j])
        else:
            boxes = boxes_new_scores

    bounding_boxes_from_key_poses = get_bounding_boxes(new_poses, pose_score)
    if len(bounding_boxes_from_key_poses) != 0:
        boxes = np.append(boxes, bounding_boxes_from_key_poses, axis=0)

    return boxes


def draw_combined(path_to_filenames, output_dir, write_score, text):
    if text:
        dir = os.path.join('op_room_evaluate', 'combined')
        if not os.path.isdir(dir):
            os.makedirs(dir)

    for image_path in path_to_filenames:

        image_name = image_path.rsplit('/', 1)[1]
        computer_node = image_path.rsplit('/', 2)[1]

        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)

        dets = bounding_box(img_raw)
        pose_preds, scores = pose(img_raw)

        # we only want the three key poses: left eye, right eye and nose
        final_boxes = final_bounding_boxes(dets, [li[:3] for li in pose_preds], scores)
        if args.save:
            for boxes in final_boxes:
                if boxes[4] < args.vis_thres:
                    continue
                text = "{:.4f}".format(boxes[4])
                boxes = list(map(int, boxes))
                cv2.rectangle(img_raw, (boxes[0], boxes[1]), (boxes[2], boxes[3]), (0, 0, 255), 2)
                if write_score:
                    cx = boxes[0]
                    cy = boxes[1] + 12
                    cv2.putText(img_raw, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

            cv2.imwrite(output_dir + image_name, img_raw)

        if text:
            dirname = os.path.join('op_room_evaluate/combined', computer_node)
            if not os.path.isdir(dirname):
                os.makedirs(dirname)

            save_name = dirname + '/' + image_name[:-4] + ".txt"

            with open(save_name, "w") as fd:
                file_name = image_name + "\n"
                final_boxes = [box for box in final_boxes if box[4] > args.vis_thres]
                bounding_box_num = str(len(final_boxes)) + "\n"
                fd.write(file_name)
                fd.write(bounding_box_num)
                for box in final_boxes:
                    x = int(box[0])
                    y = int(box[1])
                    w = int(box[2]) - int(box[0])
                    h = int(box[3]) - int(box[1])
                    confidence = str(box[4])
                    line = str(x) + " " + str(y) + " " + str(w) + " " + str(h) + " " + confidence + " \n"
                    fd.write(line)


def draw_key_poses(path_to_filenames, output_dir, write_score):
    for image_path in path_to_filenames:
        image_name = image_path.rsplit('/', 1)[1]
        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)

        pose_preds, scores = pose(img_raw)

        i = 0
        for coords in pose_preds:
            # Draw each point on image (we only want left eye, right eye and nose)
            for coord in coords[:3]:
                x_coord, y_coord = int(coord[0]), int(coord[1])
                cv2.circle(img_raw, (x_coord, y_coord), 4, (0, 0, 255), 4)

            if write_score:
                cx = int(coords[0][0])
                cy = int(coords[0][1]) + 12
                text = "{:.4f}".format(scores[i])
                cv2.putText(img_raw, text, (cx, cy),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
            i += 1

        cv2.imwrite(output_dir + image_name, img_raw)


def draw_retinaface(path_to_filenames, output_dir, write_score, text):
    if text:
        dir = os.path.join('op_room_evaluate', 'combined')
        if not os.path.isdir(dir):
            os.makedirs(dir)

    for image_path in path_to_filenames:

        image_name = image_path.rsplit('/', 1)[1]
        computer_node = image_path.rsplit('/', 2)[1]
        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)

        final_boxes = bounding_box(img_raw)
        if args.save:
            for boxes in final_boxes:
                if boxes[4] < args.vis_thres:
                    continue
                text = "{:.4f}".format(boxes[4])
                boxes = list(map(int, boxes))
                cv2.rectangle(img_raw, (boxes[0], boxes[1]), (boxes[2], boxes[3]), (0, 0, 255), 2)
                if write_score:
                    cx = boxes[0]
                    cy = boxes[1] + 12
                    cv2.putText(img_raw, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

            cv2.imwrite(output_dir + image_name, img_raw)

        if text:

            dirname = os.path.join('op_room_evaluate/retinaface', computer_node)
            if not os.path.isdir(dirname):
                os.makedirs(dirname)

            save_name = dirname + '/' + image_name[:-4] + ".txt"

            with open(save_name, "w") as fd:
                file_name = image_name + "\n"
                final_boxes = [box for box in final_boxes if box[4] > args.vis_thres]
                bounding_box_num = str(len(final_boxes)) + "\n"
                fd.write(file_name)
                fd.write(bounding_box_num)
                for box in final_boxes:
                    x = int(box[0])
                    y = int(box[1])
                    w = int(box[2]) - int(box[0])
                    h = int(box[3]) - int(box[1])
                    confidence = str(box[4])
                    line = str(x) + " " + str(y) + " " + str(w) + " " + str(h) + " " + confidence + " \n"
                    fd.write(line)


if __name__ == '__main__':
    files = glob.glob('/home/tony/Documents/CAP/media/data_op/cn03/' + '**/*.jpg', recursive=True)
    # files = files[:10]

    if args.detect == "combined":
        draw_combined(files, args.outputDir, args.write_scores, args.text)
    elif args.detect == "key_poses":
        draw_key_poses(files, args.outputDir, args.write_scores)
    else:
        draw_retinaface(files, args.outputDir, args.write_scores, args.text)
