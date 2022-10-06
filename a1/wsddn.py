import numpy as np
import torch
import torchvision
import torch.utils.data as data
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torchvision.ops import roi_pool, roi_align


class WSDDN(nn.Module):
    n_classes = 20
    classes = np.asarray([
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
        'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ])

    def __init__(self, classes=None):
        super(WSDDN, self).__init__()

        if classes is not None:
            self.classes = classes
            self.n_classes = len(classes)
            print(classes)

        # TODO (Q2.1): Define the WSDDN model
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
        )
        self.roi_pool = roi_pool
        self.classifier = nn.Sequential(
            nn.Linear(9216, 4096),
            nn.ReLU(inplace=False),
            nn.Dropout2d(inplace=False),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=False),
        )

        self.score_fc = nn.Linear(4096, 20)
        self.bbox_fc = nn.Linear(4096, 20)

        # loss
        self.cross_entropy = None

    @property
    def loss(self):
        return self.cross_entropy

    def forward(self,
                image,
                rois=None,
                gt_vec=None,
                ):

        # TODO (Q2.1): Use image and rois as input
        # compute cls_prob which are N_roi X 20 scores
        features = self.features(image)
        pooled_features = self.roi_pool(input=features, boxes=rois, output_size=(6, 6), spatial_scale=1.0 / 16 * image.size(2))
        x = pooled_features.view(pooled_features.size(0), -1)
        x = self.classifier(x)
        # shape of x should be num_roi x 4096
        cls_score = self.score_cls(x)
        cls_softmax = F.softmax(cls_score, dim=1)
        det_score = self.score_det(x)
        det_softmax = F.softmax(det_score, dim=0)
        cls_prob = cls_softmax * det_softmax

        if self.training:
            label_vec = gt_vec.view(-1, self.n_classes)
            self.cross_entropy = self.build_loss(cls_prob, label_vec)
        return cls_prob

    def build_loss(self, cls_prob, label_vec):
        """Computes the loss

        :cls_prob: N_roix20 output scores
        :label_vec: 1x20 one hot label vector
        :returns: loss

        """
        # TODO (Q2.1): Compute the appropriate loss using the cls_prob
        # that is the output of forward()
        # Checkout forward() to see how it is called
        cls_prob = torch.sum(cls_prob, 0).view(-1, self.n_classes)  # 1xc
        cls_prob = torch.clamp(cls_prob, 0, 1)
        loss = F.binary_cross_entropy(cls_prob, label_vec, size_average=False)

        return loss
