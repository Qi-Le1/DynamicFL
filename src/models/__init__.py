from .wresnet import (
    WideResNet,
    wresnet28x2,
    wresnet28x8,
    wresnet37x2
)

from .resnet import (
    ResNet,
    resnet9,
    resnet18
)

from .utils import (
    make_batchnorm,
)


__all__ = [
    'WideResNet',
    'wresnet28x2',
    'wresnet28x8',
    'wresnet37x2',
    'ResNet',
    'resnet9',
    'resnet18'
    'make_batchnorm'
]
