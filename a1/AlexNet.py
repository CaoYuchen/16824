import torch.nn as nn
import torch
import torchvision.models as models

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class LocalizerAlexNet(nn.Module):
    def __init__(self, num_classes=20):
        super(LocalizerAlexNet, self).__init__()
        # TODO (Q1.1): Define model
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
            # nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=1, stride=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 20, kernel_size=1, stride=1),
        )

    def forward(self, x):
        # TODO (Q1.1): Define forward pass
        x = self.features(x)
        x = self.classifier(x)
        return x


class LocalizerAlexNetRobust(nn.Module):
    def __init__(self, num_classes=20):
        super(LocalizerAlexNetRobust, self).__init__()
        # TODO (Q1.7): Define model
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=False),
            nn.AvgPool2d(kernel_size=3, stride=2, ceil_mode=False),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=False),
            nn.AvgPool2d(kernel_size=3, stride=2, ceil_mode=False),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            # nn.AvgPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=1, stride=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 20, kernel_size=1, stride=1),
            # nn.Dropout(p=0.3, inplace=False),
        )
    def forward(self, x):
        # TODO (Q1.7): Define forward pass
        x = self.features(x)
        x = self.classifier(x)
        return x


def localizer_alexnet(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = LocalizerAlexNet(**kwargs)
    # TODO (Q1.3): Initialize weights based on whether it is pretrained or not
    if(pretrained):
        torch.hub.set_dir("E:\\16824\\a1\\.cache")
        AlexNet = models.alexnet(pretrained=True)
        for i, layer in enumerate(model.features):
            if type(layer) == nn.Conv2d:
                layer.weight = AlexNet.features[i].weight
                layer.bias = AlexNet.features[i].bias
        for i, layer in enumerate(model.classifier):
            if type(layer) == nn.Conv2d:
                nn.init.xavier_uniform_(layer.weight)
                # nn.init.xavier_uniform_(layer.bias)
                torch.nn.init.zeros_(layer.bias)

    return model


def localizer_alexnet_robust(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = LocalizerAlexNetRobust(**kwargs)
    # TODO (Q1.7): Initialize weights based on whether it is pretrained or not
    if(pretrained):
        torch.hub.set_dir("E:\\16824\\a1\\.cache")
        AlexNet = models.alexnet(pretrained=True)
        for i, layer in enumerate(model.features):
            if type(layer) == nn.Conv2d:
                layer.weight = AlexNet.features[i].weight
                layer.bias = AlexNet.features[i].bias
        for i, layer in enumerate(model.classifier):
            if type(layer) == nn.Conv2d:
                nn.init.xavier_uniform_(layer.weight)
                # nn.init.xavier_uniform_(layer.bias)
                torch.nn.init.zeros_(layer.bias)

    return model
