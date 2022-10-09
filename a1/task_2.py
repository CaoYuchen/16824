from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import argparse
import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
import numpy as np
from datetime import datetime
import pickle as pkl

# imports
from wsddn import WSDDN
from voc_dataset import *
import wandb
from utils import nms, iou, tensor_to_PIL
from PIL import Image, ImageDraw

from sklearn.metrics import auc

# hyper-parameters
# ------------
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument(
    '--lr',
    default=0.0001,
    type=float,
    help='Learning rate'
)
parser.add_argument(
    '--lr-decay-steps',
    default=150000,
    type=int,
    help='Interval at which the lr is decayed'
)
parser.add_argument(
    '--lr-decay',
    default=0.1,
    type=float,
    help='Decay rate of lr'
)
parser.add_argument(
    '--momentum',
    default=0.9,
    type=float,
    help='Momentum of optimizer'
)
parser.add_argument(
    '--weight-decay',
    default=0.0005,
    type=float,
    help='Weight decay'
)
parser.add_argument(
    '--epochs',
    default=5,
    type=int,
    help='Number of epochs'
)
parser.add_argument(
    '--val-interval',
    default=5000,
    type=int,
    help='Interval at which to perform validation'
)
parser.add_argument(
    '--disp-interval',
    default=10,
    type=int,
    help='Interval at which to perform visualization'
)
parser.add_argument(
    '--use-wandb',
    default=False,
    type=bool,
    help='Flag to enable visualization'
)
parser.add_argument(
    '-j',
    '--workers',
    default=4,
    type=int,
    metavar='N',
    help='number of data loading workers (default: 4)')
parser.add_argument(
    '-b',
    '--batch-size',
    default=1,  # 256
    type=int,
    metavar='N',
    help='mini-batch size (default: 256)')
# ------------

# Set random seed
rand_seed = 1024
if rand_seed is not None:
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)

# Set output directory
output_dir = "./"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def calculate_map(gt_boxes, gt_class_list, pred_boxes, pred_scores, iou_thresh=0.3):
    """
    Calculate the mAP for classification.
    """
    # TODO (Q2.3): Calculate mAP on test set.
    # Feel free to write necessary function parameters.
    precisions = []
    recalls = []

    FP = 0
    TP = 0
    gt_class_used = np.zeros_like(gt_class_list)
    for class_num in range(20):
        # check if gt_class_list contains the class_num, otherwise FP++
        if class_num not in gt_class_list:
            FP += 1
        else:
            # iterate through gt_class_list and compare IOU with nms_box to decide TP and FP
            for i, gt_class in enumerate(gt_class_list):
                if gt_class_used[i] == 1:
                    continue
                gt_box = gt_boxes[gt_class]
                for pred_box in pred_boxes:
                    if iou(gt_box, pred_box) >= iou_thresh:
                        TP += 1
                        gt_class_used[i] = 1
                    else:
                        FP += 1

    precisions.append(TP / (TP + FP))
    recalls.append(TP / gt_class_list.shape[0])

    mAP = auc(recalls,precisions)
    return mAP


def test_model(model, val_loader=None, thresh=0.0002):  # 0.05
    """
    Tests the networks and visualizes the detections
    :param thresh: Confidence threshold
    """
    with torch.no_grad():
        for iter, data in enumerate(val_loader):

            # one batch = data for one image
            image = data['image'].cuda()
            target = data['label'].cuda()
            # wgt = data['wgt'].cuda()
            rois = data['rois'].cuda()
            gt_boxes = data['gt_boxes'].squeeze().numpy()
            gt_class_list = data['gt_classes'].squeeze().numpy()

            # TODO (Q2.3): perform forward pass, compute cls_probs
            cls_probs = model(image, rois, target)
            cls_probs = cls_probs.data.cpu().numpy()
            rois = rois.data.cpu().numpy()
            # TODO (Q2.3): Iterate over each class (follow comments)
            pred_boxes = []
            pred_scores = []
            # gt_boxes_list = []
            iou_thresh = 0.3
            for class_num in range(20):
                # get valid rois and cls_scores based on thresh
                index = np.where(cls_probs[:, class_num] > thresh)[0]
                scores = cls_probs[index, class_num]
                boxes = rois[0, index]
                # use NMS to get boxes and scores
                nms_boxes, nms_scores = nms(boxes, scores, threshold=iou_thresh)
                pred_boxes.append(nms_boxes)
                pred_scores.append(nms_scores)
                # if class_num in gt_class_list:
                #     i = np.where(gt_class_list == class_num)[0]
                #     gt_boxes_list.append(gt_boxes[i])
                # else:
                #     gt_boxes_list.append(None)

        # TODO (Q2.3): visualize bounding box predictions when required
        mAP = calculate_map(gt_boxes, gt_class_list, pred_boxes, pred_scores, iou_thresh=iou_thresh)


def train_model(model, train_loader=None, val_loader=None, optimizer=None, args=None):
    """
    Trains the network, runs evaluation and visualizes the detections
    """
    # Initialize training variables
    train_loss = 0
    step_cnt = 0
    for epoch in range(args.epochs):
        for iter, data in enumerate(train_loader):

            # TODO (Q2.2): get one batch and perform forward pass
            # one batch = data for one image
            image = data['image'].cuda()
            target = data['label'].cuda()
            # wgt = data['wgt'].cuda()
            rois = data['rois'].cuda()
            # gt_boxes = data['gt_boxes'].squeeze().numpy()
            # gt_class_list = data['gt_classes'].squeeze().numpy()

            # TODO (Q2.2): perform forward pass
            # take care that proposal values should be in pixels
            # Convert inputs to cuda if training on GPU

            model(image, rois, target)

            # backward pass and update
            loss = model.loss
            train_loss += loss.item()
            step_cnt += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # TODO (Q2.2): evaluate the model every N iterations (N defined in handout)
            # Add wandb logging wherever necessary
            ap = test_model(model, val_loader)
            if iter % args.val_interval == 0 and iter != 0:
                model.eval()
                ap = test_model(model, val_loader)
                print("AP ", ap)
                model.train()

            # TODO (Q2.4): Perform all visualizations here
            # The intervals for different things are defined in the handout

    # TODO (Q2.4): Plot class-wise APs


def main():
    """
    Creates dataloaders, network, and calls the trainer
    """
    args = parser.parse_args()
    # TODO (Q2.2): Load datasets and create dataloaders
    # Initialize wandb logger
    train_dataset = VOCDataset(split='trainval',
                               image_size=512,
                               top_n=300,
                               data_dir='data/train/VOCdevkit/VOC2007/')
    val_dataset = VOCDataset(split='test',
                             image_size=512,
                             top_n=300,
                             data_dir='data/test/VOCdevkit/VOC2007/')
    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # true
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
        collate_fn=VOCDataset.custom_collate)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=VOCDataset.custom_collate)
    # Create network and initialize
    net = WSDDN(classes=train_dataset.CLASS_NAMES)
    print(net)

    alex_path = "E:\\16824\\a1\\.cache\\pretrained_alexnet.pkl"
    if os.path.exists(alex_path):
        pret_net = pkl.load(open(alex_path, 'rb'))
    else:
        pret_net = model_zoo.load_url(
            'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth')
        pkl.dump(pret_net,
                 open(alex_path, 'wb'), pkl.HIGHEST_PROTOCOL)
    own_state = net.state_dict()

    for name, param in pret_net.items():
        print(name)
        if name not in own_state:
            continue
        if isinstance(param, Parameter):
            param = param.data
        try:
            own_state[name].copy_(param)
            print('Copied {}'.format(name))
        except:
            print('Did not find {}'.format(name))
            continue

    # Move model to GPU and set train mode
    net.load_state_dict(own_state)
    net.cuda()
    net.train()

    # TODO (Q2.2): Freeze AlexNet layers since we are loading a pretrained model
    for param in net.features.parameters():
        param.requires_grad = False
    # TODO (Q2.2): Create optimizer only for network parameters that are trainable
    optimizer = torch.optim.SGD(net.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    # Training
    train_model(net, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer, args=args)


if __name__ == '__main__':
    main()
