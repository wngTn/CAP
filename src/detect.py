from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import print_function

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
parser.add_argument('--cfg', default='../experiments/coco/w48/w48_4x_reg03_bs5_640_adam_lr1e-3_coco_x140.yaml',
                    type=str)
parser.add_argument('--outputDir', type=str, default='/output/')
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
    pose_model.load_state_dict(torch.load(
        '../model/pose_coco/pose_dekr_hrnetw48_coco.pth'), strict=False)
    # else:
    # raise ValueError('expected model defined in config at TEST.MODEL_FILE')

    pose_model.to(CTX)
    pose_model.eval()

    image_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
    image_pose = image_rgb.copy()
    pose_preds = get_pose_estimation_prediction(cfg, pose_model, image_pose, 0.3, transforms=pose_transform)

    return pose_preds


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


def boundingbox(img_raw):
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


def isinboxes(boxes, x_avg, y_avg, score):
    buffer = 5
    for j in range(len(boxes)):
        b = boxes[j]
        [x_least, y_least, x_max, y_max, _] = b
        if ((x_least - buffer) <= x_avg <= (x_max + buffer)) and ((y_least - buffer) <= y_avg <= (y_max + buffer)):
            boxes[j][4] += (score * 1.5)
            return boxes, True

    return boxes, False


def getbbox(poses, pose_score):
    result = []
    for j in range(len(poses)):
        coords = poses[j]
        x_values = [tup[0] for tup in coords]
        y_values = [tup[1] for tup in coords]
        x_least = min(x_values) - 5
        x_max = max(x_values) + 5
        y_least = min(y_values) - 10
        y_max = max(y_values) + 5
        result.append([x_least, y_least, x_max, y_max, pose_score[j]])
    return result


def finalboundbox(boxes, poses, pose_score):
    new_poses = []
    new_scores = []
    for j in range(len(poses)):
        coords = poses[j]
        x_avg = mean([tup[0] for tup in coords])
        y_avg = mean([tup[1] for tup in coords])
        newBoxes, bool = isinboxes(boxes, x_avg, y_avg, pose_score[j])
        if not bool:
            new_poses.append(coords)
            new_scores.append(pose_score[j])
        else:
            boxes = newBoxes

    poseboxes = getbbox(new_poses, pose_score)
    if len(poseboxes) != 0:
        boxes = np.append(boxes, poseboxes, axis=0)

    return boxes


if __name__ == '__main__':
    # image_path = "/home/tony/Documents/CAP/media/not_annotated/cn03/0000008190_color_annotated.jpg"
    pose_preds = []
    scores = []

    imgdir = '/home/tony/Documents/CAP/media/data_op/cn04/'

    files = os.listdir(imgdir)

    files = files[10:]

    for image in files:
        image_path = imgdir + image
        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)

        dets = boundingbox(img_raw)
        pose_preds, scores = pose(img_raw)

        finalbboxes = finalboundbox(dets, [li[:3] for li in pose_preds], scores)

        # i = 0
        # for coords in pose_preds:
        #     # Draw each point on image
        #     for coord in coords[:3]:
        #         x_coord, y_coord = int(coord[0]), int(coord[1])
        #         cv2.circle(img_raw, (x_coord, y_coord), 4, (0, 0, 255), 2)
        #
        #     cx = int(coords[0][0])
        #     cy = int(coords[0][1]) + 12
        #     text = "{:.4f}".format(scores[i])
        #     cv2.putText(img_raw, text, (cx, cy),
        #                 cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
        #     i += 1

        # show image
        for b in finalbboxes:
            if b[4] < args.vis_thres:
                continue
            # text = "{:.4f}".format(b[4])
            b = list(map(int, b))
            cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cx = b[0]
            cy = b[1] + 12
            # cv2.putText(img_raw, text, (cx, cy),
            #             cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

        name = "./output/cn04/"
        cv2.imwrite(name + image, img_raw)
