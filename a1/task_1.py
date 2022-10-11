import argparse
import os
import shutil
import time

import sklearn
import sklearn.metrics
from sklearn.metrics import fbeta_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import wandb

import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import cm
import numpy as np

from AlexNet import localizer_alexnet, localizer_alexnet_robust
from voc_dataset import *
from utils import *

from torch.optim.lr_scheduler import StepLR

USE_WANDB = False  # use flags, wandb is not convenient for debugging
USE_WANDB_IMAGE = False
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', default='localizer_alexnet')
parser.add_argument(
    '-j',
    '--workers',
    default=4,
    type=int,
    metavar='N',
    help='number of data loading workers (default: 4)')
parser.add_argument(
    '--epochs',
    default=30,  # 30
    type=int,
    metavar='N',
    help='number of total epochs to run')
parser.add_argument(
    '--start-epoch',
    default=0,
    type=int,
    metavar='N',
    help='manual epoch number (useful on restarts)')
parser.add_argument(
    '-b',
    '--batch-size',
    default=32,  # 256
    type=int,
    metavar='N',
    help='mini-batch size (default: 256)')
parser.add_argument(
    '--lr',
    '--learning-rate',
    default=0.1,
    type=float,
    metavar='LR',
    help='initial learning rate')
parser.add_argument(
    '--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument(
    '--weight-decay',
    '--wd',
    default=1e-4,
    type=float,
    metavar='W',
    help='weight decay (default: 1e-4)')
parser.add_argument(
    '--print-freq',
    '-p',
    default=10,
    type=int,
    metavar='N',
    help='print frequency (default: 10)')
parser.add_argument(
    '--eval-freq',
    default=2,
    type=int,
    metavar='N',
    help='print frequency (default: 10)')
parser.add_argument(
    '--resume',
    default='',
    type=str,
    metavar='PATH',
    help='path to latest checkpoint (default: none)')
parser.add_argument(
    '-e',
    '--evaluate',
    dest='evaluate',
    action='store_true',
    help='evaluate model on validation set')
parser.add_argument(
    '--pretrained',
    dest='pretrained',
    action='store_false',
    help='use pre-trained model')
parser.add_argument(
    '--world-size',
    default=1,
    type=int,
    help='number of distributed processes')
parser.add_argument(
    '--dist-url',
    default='tcp://224.66.41.62:23456',
    type=str,
    help='url used to set up distributed training')
parser.add_argument(
    '--dist-backend', default='gloo', type=str, help='distributed backend')
parser.add_argument('--vis', action='store_true')
parser.add_argument('--avg-pool', default=False, type=bool, help="Use avg_pool instead of max_pool")

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()
    args.distributed = args.world_size > 1

    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.arch == 'localizer_alexnet':
        model = localizer_alexnet(pretrained=args.pretrained)
    elif args.arch == 'localizer_alexnet_robust':
        model = localizer_alexnet_robust(pretrained=args.pretrained)
    print(model)

    model.features = torch.nn.DataParallel(model.features)
    model.cuda()

    # TODO (Q1.1): define loss function (criterion) and optimizer from [1]
    # also use an LR scheduler to decay LR by 10 every 30 epochs
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code

    # TODO (Q1.1): Create Datasets and Dataloaders using VOCDataset
    # Ensure that the sizes are 512x512
    # Also ensure that data directories are correct
    # The ones use for testing by TAs might be different
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
        shuffle=False,
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

    if args.evaluate:
        validate(val_loader, model, criterion, wandb=wandb)
        return

    # TODO (Q1.3): Create loggers for wandb.
    # Ideally, use flags since wandb makes it harder to debug code.
    if USE_WANDB:
        wandb.init(project="vlr-hw1", reinit=True)
        # wandb.watch(model, log_freq=100)
        wandb.define_metric("train/step")
        wandb.define_metric("train/*", step_metric="train/step")

        wandb.define_metric("val/epoch")
        wandb.define_metric("val/*", step_metric="val/epoch")

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, wandb, args.avg_pool)

        # evaluate on validation set
        if epoch % args.eval_freq == 0 or epoch == args.epochs - 1:
            m1, m2 = validate(val_loader, model, criterion, epoch, wandb, args.avg_pool)

            score = m1 * m2
            # remember best prec@1 and save checkpoint
            is_best = score > best_prec1
            best_prec1 = max(score, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best)

        scheduler.step()


# TODO: You can add input arguments if you wish
def train(train_loader, model, criterion, optimizer, epoch, wandb, avg_pool=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    avg_m1 = AverageMeter()
    avg_m2 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (data) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # TODO (Q1.1): Get inputs from the data dict
        # Convert inputs to cuda if training on GPU
        target = data["label"].cuda()
        input = data["image"].cuda()

        # TODO (Q1.1): Get output from model
        imoutput = model(input)

        # TODO (Q1.1): Perform any necessary operations on the output
        # maxPool the batch*channel*n*m to batch*channel*1*1 for global label
        if avg_pool is True:
            avgPool = nn.AvgPool2d((imoutput.size(dim=2), imoutput.size(dim=3)), stride=1)
            output = avgPool(imoutput).flatten(start_dim=1)
        else:
            maxPool = nn.MaxPool2d((imoutput.size(dim=2), imoutput.size(dim=3)), stride=1)
            output = maxPool(imoutput).flatten(start_dim=1)
        output = torch.sigmoid(output)
        # TODO (Q1.1): Compute loss using ``criterion``
        loss = criterion(output, target)

        # measure metrics and record loss
        m1 = metric1(output.data, target)
        m2 = metric2(output.data, target)
        losses.update(loss.item(), input.size(0))
        avg_m1.update(m1)
        avg_m2.update(m2)

        # TODO (Q1.1): compute gradient and perform optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Metric1 {avg_m1.val:.3f} ({avg_m1.avg:.3f})\t'
                  'Metric2 {avg_m2.val:.3f} ({avg_m2.avg:.3f})'.format(
                epoch,
                i,
                len(train_loader),
                batch_time=batch_time,
                data_time=data_time,
                loss=losses,
                avg_m1=avg_m1,
                avg_m2=avg_m2))

        # TODO (Q1.3): Visualize/log things as mentioned in handout at appropriate intervals

        if USE_WANDB:
            wandb.log({'train/step': epoch * len(train_loader) + i})
            wandb.log({'train/loss': losses.val})
            wandb.log({'train/M1': avg_m1.val})
            wandb.log({'train/M2': avg_m2.val})
            wandb.log({'train/LR': optimizer.param_groups[0]['lr']})

        epoch_to_plot = [0, 14, 29]
        batch_to_plot = [10, 69]
        index_image = 5
        if USE_WANDB_IMAGE and epoch in epoch_to_plot and i in batch_to_plot:
            index_class = (target.flatten() == 1).nonzero().flatten().tolist()[0]
            heatmap = torch.sigmoid(
                F.interpolate(imoutput[index_image:index_image + 1, index_class:index_class + 1],
                              [input.size(2), input.size(3)],
                              mode='bilinear', align_corners=True))
            heatmap = Image.fromarray(np.uint8(cm.jet(heatmap[0][0].cpu().detach().numpy()) * 255))
            # heatmap = torch.cat((heatmap, heatmap, heatmap), 1)
            # heatmap = tensor_to_PIL(heatmap[0])
            gt_image = wandb.Image(tensor_to_PIL(input[index_image]),
                                   caption='RGB Image')
            heatmap = wandb.Image(heatmap, caption=f'{VOCDataset.CLASS_NAMES[index_class]}')
            wandb.log({"train/Heatmaps": [gt_image, heatmap]})

        # End of train()


def validate(val_loader, model, criterion, epoch=0, wandb=None, avg_pool=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_m1 = AverageMeter()
    avg_m2 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (data) in enumerate(val_loader):

        # TODO (Q1.1): Get inputs from the data dict
        # Convert inputs to cuda if training on GPU
        target = data["label"].cuda()
        input = data["image"].cuda()

        # TODO (Q1.1): Get output from model
        imoutput = model(input)

        # TODO (Q1.1): Perform any necessary functions on the output
        # maxPool the batch*channel*n*m to batch*channel*1*1 for global label
        if avg_pool is True:
            avgPool = nn.AvgPool2d((imoutput.size(dim=2), imoutput.size(dim=3)), stride=1)
            output = avgPool(imoutput).flatten(start_dim=1)
        else:
            maxPool = nn.MaxPool2d((imoutput.size(dim=2), imoutput.size(dim=3)), stride=1)
            output = maxPool(imoutput).flatten(start_dim=1)
        output = torch.sigmoid(output)
        # TODO (Q1.1): Compute loss using ``criterion``
        loss = criterion(output, target)

        # measure metrics and record loss
        m1 = metric1(output.data, target)
        m2 = metric2(output.data, target)
        losses.update(loss.item(), input.size(0))
        avg_m1.update(m1)
        avg_m2.update(m2)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Metric1 {avg_m1.val:.3f} ({avg_m1.avg:.3f})\t'
                  'Metric2 {avg_m2.val:.3f} ({avg_m2.avg:.3f})'.format(
                i,
                len(val_loader),
                batch_time=batch_time,
                loss=losses,
                avg_m1=avg_m1,
                avg_m2=avg_m2))


        # TODO (Q1.3): Visualize things as mentioned in handout
        # TODO (Q1.3): Visualize at appropriate intervals
        epoch_to_plot = 29
        batch_to_plot = [5, 21, 57]
        index_image = 3
        if USE_WANDB_IMAGE and epoch == epoch_to_plot and i in batch_to_plot:
            index_class = (target.flatten() == 1).nonzero().flatten().tolist()[0]
            heatmap = torch.sigmoid(
                F.interpolate(imoutput[index_image:index_image + 1, index_class:index_class + 1],
                              [input.size(2), input.size(3)],
                              mode='bilinear', align_corners=True))
            heatmap = Image.fromarray(np.uint8(cm.jet(heatmap[0][0].cpu().detach().numpy()) * 255))
            # heatmap = torch.cat((heatmap, heatmap, heatmap), 1)
            # heatmap = tensor_to_PIL(heatmap[0])
            gt_image = wandb.Image(tensor_to_PIL(input[index_image]),
                                   caption='RGB Image')
            heatmap = wandb.Image(heatmap, caption=f'{VOCDataset.CLASS_NAMES[index_class]}')
            wandb.log({"val/Heatmaps": [gt_image, heatmap]})
        # heatmap = torch.cat((heatmap, heatmap, heatmap), 1)
        # heatmap = tensor_to_PIL(heatmap[0])

    # print avg in an epoch fashion
    if USE_WANDB and epoch % args.eval_freq == 0:
        # step = epoch // args.eval_freq * len(val_loader) + i
        wandb.log({'val/epoch': epoch})
        # wandb.log({'val/loss': losses.val})
        wandb.log({'val/M1': avg_m1.avg})
        wandb.log({'val/M2': avg_m2.avg})

    print(' * Metric1 {avg_m1.avg:.3f} Metric2 {avg_m2.avg:.3f}'.format(
        avg_m1=avg_m1, avg_m2=avg_m2))

    return avg_m1.avg, avg_m2.avg


# TODO: You can make changes to this function if you wish (not necessary)
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def metric1(output, target, threshold=0.5):
    # TODO (Q1.5): compute metric1
    n_class = target.shape[1]
    sum_ap = []
    for i in range(n_class):
        gt_class = target[:, i].cpu().numpy().astype('float32')
        pred_class = output[:, i].cpu().numpy().astype('float32')
        # for cases that there is no gt classes, we ignore the predict 0, and count non-zero predict as ap = 0
        if np.count_nonzero(gt_class) == 0:
            if np.count_nonzero(pred_class > threshold) == 0:
                continue
            else:
                ap = 0
        else:
            ap = average_precision_score(gt_class, pred_class)
        sum_ap.append(ap)
    mAP = np.mean(sum_ap)
    return mAP


def metric2(output, target, threshold=0.5):
    # TODO (Q1.5): compute metric2
    n_class = target.shape[1]
    sum_recall = []
    for i in range(n_class):
        gt_class = target[:, i].cpu().numpy().astype('float32')
        pred_class = output[:, i].cpu().numpy().astype('float32')
        pred_class = (pred_class > threshold).astype(int)
        if np.count_nonzero(gt_class) == 0:
            continue
        else:
            recall = recall_score(gt_class, pred_class, average="samples")
        sum_recall.append(recall)
    m_recall = np.mean(sum_recall)
    return m_recall


if __name__ == '__main__':
    main()
