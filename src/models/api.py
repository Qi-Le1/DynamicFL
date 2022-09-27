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

# from .cnn import CNN

# from .generator import Generator

from .utils import (
    make_batchnorm,
)

def create_model(track_running_stats=False):
    from config import cfg
    if cfg['model_name'] == 'resnet9':
        model = resnet9()
    else:
        raise ValueError('model_name is wrong')
        
    model.to(cfg["device"])
    model.apply(lambda m: make_batchnorm(m, momentum=None, track_running_stats=track_running_stats))
    return model

__all__ = [
    'WideResNet',
    'wresnet28x2',
    'wresnet28x8',
    'wresnet37x2',
    'ResNet',
    'resnet9',
    'resnet18',
    # 'CNN',
    # 'Generator'
    'make_batchnorm',
    'create_model'
]
