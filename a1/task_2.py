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
from utils import nms, iou, tensor_to_PIL, get_box_data_caption
from PIL import Image, ImageDraw

from sklearn.metrics import auc
from torch.optim.lr_scheduler import StepLR
from task_1 import AverageMeter

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
    default=1,  # 1
    type=int,
    metavar='N',
    help='mini-batch size (default: 256)')
# ------------
# wandb related parameters
USE_WANDB = False
USE_WANDB_IMAGE = False
class_id_to_label = dict(enumerate(VOCDataset.CLASS_NAMES))
images_to_plot = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500]
epoch_to_plot = [0, 4]
# Set random seed
rand_seed = 1024
if rand_seed is not None:
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)

# Set output directory
output_dir = "./"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def calculate_map(all_bboxes, all_scores, all_batches, all_gt_boxes, all_gt_classes, iou_thresh=0.3, n_classes=20,
                  eps=1e-5):
    """
    Calculate the mAP for classification.
    """
    # TODO (Q2.3): Calculate mAP on test set.
    # Feel free to write necessary function parameters.
    ## For all_bboxes, all_scores, all_batches
    # The list size is C x N x Array(WxH or S)
    # N is number of batches multiply by number of predicted bboxes, C is the number of classes
    # W x H is for box size, S is for and scores size and batches index

    ## For all_gt_boxes, all_gt_classes
    # The list is N x M x Array(WxH or S)
    # N is number of batches, M is the number of bboxes for each batch
    # W x H is for box size, S is for and scores size and batches index

    aps = []
    # per class
    for n_class in range(n_classes):
        tp = 0
        fp = 0
        precisions = []
        recalls = []
        gts_count = all_gt_classes[n_class].size(0)

        orders = np.asarray(-all_scores[n_classes]).argsort()
        visited = np.zeros_like(all_gt_classes[n_class])
        visited_count = 0
        # per predicted bbox
        for order in orders:
            if visited_count == visited.size(0):
                fp += 1
                precisions.append(tp / (tp + fp + eps))
                recalls.append(tp / (gts_count + eps))
                continue

            bbox = all_bboxes[n_class][order]
            batch = all_batches[n_class][order]
            gt_boxes = all_gt_boxes[batch]
            gt_classes = all_gt_classes[batch]

            if n_class not in gt_classes:
                fp += 1
                precisions.append(tp / (tp + fp + eps))
                recalls.append(tp / (gts_count + eps))
            else:
                indices = np.where(gt_classes == n_class)[0]
                for i in indices:
                    if visited[i]:
                        continue
                    else:
                        iou_score = iou(bbox, gt_boxes[i])
                        if iou_score >= iou_thresh:
                            tp += 1
                            visited[i] = 1
                            visited_count += 1
                        else:
                            fp += 1
                        precisions.append(tp / (tp + fp + eps))
                        recalls.append(tp / (gts_count + eps))

        # Calculating auc.
        ap = auc(np.asarray(recalls), np.asarray(precisions))
        aps.append(ap)

    m_ap = np.mean(aps)
    return m_ap, aps


def test_model(model, val_loader=None, thresh=0.05, wandb=None):  # 0.05
    """
    Tests the networks and visualizes the detections
    :param thresh: Confidence threshold
    """
    # The list size is C x N x Array(WxH or S)
    # N is number of batches multiply by number of predicted bboxes, C is the number of classes
    # W x H is for box size, S is for and scores size and batches index
    all_bboxes = [[]] * 20
    all_scores = [[]] * 20
    all_batches = [[]] * 20
    # The list is N x M x Array(WxH or S)
    # N is number of batches, M is the number of bboxes for each batch
    # W x H is for box size, S is for and scores size and batches index
    all_gt_boxes = []
    all_gt_classes = []

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
            all_gt_boxes.append(gt_boxes)
            all_gt_classes.append(gt_class_list)

            pred_boxes = []
            pred_scores = []
            pred_index = []

            iou_thresh = 0.3
            first_n = 3
            for class_num in range(20):
                # get valid rois and cls_scores based on thresh
                index = np.where(cls_probs[:, class_num] > thresh)[0]
                scores = cls_probs[index, class_num]
                boxes = rois[0, index]
                # use NMS to get boxes and scores
                nms_boxes, nms_scores = nms(boxes, scores, threshold=iou_thresh)

                # append index of the class to the list together with boxes and scores for wandb plot
                pred_boxes.append(nms_boxes[0:first_n])
                pred_scores.append(nms_scores[0:first_n])
                pred_index.append(np.ones_like(nms_scores[0:first_n]) * class_num)

                # extend list for map calculation
                # all_classes.extend((np.ones_like(nms_scores) * class_num).tolist())
                all_bboxes[class_num].extend(nms_boxes)
                all_scores[class_num].extend(nms_scores)
                all_batches[class_num].extend((np.ones_like(nms_scores) * iter).tolist())

            # all_bboxes.append(pred_boxes)
            # all_scores.append(pred_scores)
            # all_classes.append(pred_index)

            # TODO (Q2.3): visualize bounding box predictions when required
            if USE_WANDB_IMAGE:
                rois_image = wandb.Image(image.cpu().detach(),
                                         boxes={
                                             "predictions": {
                                                 "box_data": get_box_data_caption(pred_index,
                                                                                  pred_boxes,
                                                                                  pred_scores,
                                                                                  VOCDataset.CLASS_NAMES),
                                                 "class_labels": class_id_to_label,
                                             },
                                         })
                wandb.log({f"val/Bounding Boxes_{iter}": rois_image})

        map, aps = calculate_map(all_bboxes, all_scores, all_batches, all_gt_boxes, all_gt_classes,
                                 iou_thresh=iou_thresh)
    return map, aps


def train_model(model, train_loader=None, val_loader=None, optimizer=None, args=None, wandb=None):
    """
    Trains the network, runs evaluation and visualizes the detections
    """
    # Initialize training variables
    train_loss = AverageMeter()
    step_cnt = 0
    loss_interval = 500
    map = 0
    aps = []
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

            cls_probs = model(image, rois, target)
            cls_probs = cls_probs.data.cpu().numpy()
            # backward pass and update
            loss = model.loss
            # train_loss += loss.item()
            train_loss.update(loss.item(), image.size(0))
            step_cnt += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # TODO (Q2.2): evaluate the model every N iterations (N defined in handout)
            # Add wandb logging wherever necessary
            # map, aps = test_model(model, val_loader, wandb=wandb)
            if iter % args.val_interval == 0 and iter != 0:
                model.eval()
                map, aps = test_model(model, val_loader, wandb=wandb)
                print("AP ", aps)
                model.train()

            # TODO (Q2.4): Perform all visualizations here
            # The intervals for different things are defined in the handout
            if USE_WANDB and iter % loss_interval == 0:
                wandb.log({'train/step': step_cnt})
                wandb.log({'train/loss': train_loss.val})
            if iter in images_to_plot and epoch in epoch_to_plot and USE_WANDB_IMAGE:
                conf_thresh = 0.05
                first_n = 3
                pred_boxes = []
                pred_scores = []
                pred_index = []

                for class_num in range(20):
                    # get valid rois and cls_scores based on thresh
                    index = np.where(cls_probs[:, class_num] > conf_thresh)[0]
                    scores = cls_probs[index, class_num]
                    boxes = rois[0, index]
                    # use NMS to get boxes and scores
                    nms_boxes, nms_scores = nms(boxes, scores)

                    pred_boxes.append(nms_boxes[0:first_n])
                    pred_scores.append(nms_scores[0:first_n])
                    pred_index.append(np.ones_like(nms_scores[0:first_n]) * class_num)

                rois_image = wandb.Image(image.cpu().detach(),
                                         boxes={
                                             "predictions": {
                                                 "box_data": get_box_data_caption(pred_index,
                                                                                  pred_boxes,
                                                                                  pred_scores,
                                                                                  VOCDataset.CLASS_NAMES),
                                                 "class_labels": class_id_to_label,
                                             },
                                         })

                wandb.log({f"train/Bounding Boxes_{iter}": rois_image})

    # TODO (Q2.4): Plot class-wise APs
    if USE_WANDB:
        wandb.log({'val/map': map})
        for i in range(5):
            wandb.log({f'val/ap_{i}_{VOCDataset.CLASS_NAMES[i]}': aps[i]})


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

    # wandb for visualization
    if USE_WANDB:
        wandb.init(project='vlr-hw2')
        wandb.watch(net, log_freq=2000)
        wandb.define_metric("train/step")
        wandb.define_metric("train/*", step_metric="train/step")
    # Training
    train_model(net, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer, args=args, wandb=wandb)


if __name__ == '__main__':
    main()
