import copy
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


# TODO: given bounding boxes and corresponding scores, perform non max suppression
def nms(bounding_boxes, confidence_score, threshold=0.0002):
    """
    bounding boxes of shape     Nx4
    confidence scores of shape  N
    threshold: confidence threshold for boxes to be considered

    return: list of bounding boxes and scores
    """
    boxes, scores = [], []
    x1 = bounding_boxes[:, 0]
    y1 = bounding_boxes[:, 1]
    x2 = bounding_boxes[:, 2]
    y2 = bounding_boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = (-confidence_score).argsort()

    while order.size > 0:
        i = order[0]
        boxes.append(bounding_boxes[i])
        scores.append(confidence_score[i])

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= threshold)[0]
        order = order[inds]

    return boxes, scores


# TODO: calculate the intersection over union of two boxes
def iou(box1, box2):
    """
    Calculates Intersection over Union for two bounding boxes (xmin, ymin, xmax, ymax)
    returns IoU vallue
    """
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    intersection_box = [max(box1[0], box2[0]),
                        max(box1[1], box2[1]),
                        min(box1[2], box2[2]),
                        min(box1[3], box2[3])]

    intersection = (intersection_box[2] - intersection_box[0]) * \
        (intersection_box[3] - intersection_box[1])

    if intersection_box[0] > intersection_box[2] or \
        intersection_box[1] > intersection_box[3]:
        intersection = 0

    union = area1 + area2 - intersection

    iou = intersection / union

    return iou


def tensor_to_PIL(image):
    """
    converts a tensor normalized image (imagenet mean & std) into a PIL RGB image
    will not work with batches (if batch size is 1, squeeze before using this)
    """
    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.255],
    )

    inv_tensor = inv_normalize(image)
    inv_tensor = torch.clamp(inv_tensor, 0, 1)
    original_image = transforms.ToPILImage()(inv_tensor).convert("RGB")

    return original_image


def get_box_data(classes, bbox_coordinates):
    """
    classes : tensor containing class predictions/gt
    bbox_coordinates: tensor containing [[xmin0, ymin0, xmax0, ymax0], [xmin1, ymin1, ...]] (Nx4)

    return list of boxes as expected by the wandb bbox plotter
    """
    box_list = [{
        "position": {
            "minX": bbox_coordinates[i][0],
            "minY": bbox_coordinates[i][1],
            "maxX": bbox_coordinates[i][2],
            "maxY": bbox_coordinates[i][3],
        },
        "class_id": classes[i],
    } for i in range(len(classes))
    ]

    return box_list
