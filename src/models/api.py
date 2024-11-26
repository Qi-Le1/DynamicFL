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

from .cnn import create_CNN

from .generator import Generator

from .utils import (
    make_batchnorm,
)

def create_model(track_running_stats=False, on_cpu=False):
    from config import cfg
    if cfg['model_name'] == 'resnet9':
        model = resnet9()
    elif cfg['model_name'] == 'resnet18':
        model = resnet18()
    elif cfg['model_name'] == 'cnn':
        model = create_CNN()
    else:
        raise ValueError('model_name is wrong')
    
    model.to(cfg["device"])

    if on_cpu:
        model.to('cpu')
    model.apply(lambda m: make_batchnorm(m, momentum=None, track_running_stats=track_running_stats))
    return model

def create_generative_model(dataset_name, embedding=False):
    from config import cfg
    model = Generator(dataset_name, embedding=embedding, latent_layer_idx=-1)
    model.to(cfg["device"])
    return model


__all__ = [
    'WideResNet',
    'wresnet28x2',
    'wresnet28x8',
    'wresnet37x2',
    'ResNet',
    'resnet9',
    'resnet18',
    # 'SimpleCNN'
    # 'CNN',
    # 'Generator'
    'make_batchnorm',
    'create_model',
    'create_generative_model'
]
