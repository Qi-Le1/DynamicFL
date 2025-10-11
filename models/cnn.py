import torch.nn as nn
import torch.nn.functional as F
# from ..utils.api import CONFIGS_
from .utils import init_param, make_batchnorm, loss_fn, apply_norm

from .utils import (
    Conv2dCosineNorm,
    FFNCosineNorm,
    Conv2dPCCNorm,
    FFNPCCNorm,
)

from config import cfg
import collections

#################################
##### Neural Network model #####
#################################
class SimpleCNNMNIST(nn.Module):
    def __init__(self, input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=62):
        super(SimpleCNNMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        # if cfg['norm'] == 'ln':
        self.n1 = nn.GroupNorm(1, 6)
        # elif cfg['norm'] == 'cn':
        #     self.conv1 = Conv2dCosineNorm(1, 6, 5)
        # elif cfg['norm'] == 'pccn':
        #     self.conv1 = Conv2dPCCNorm(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # if cfg['norm'] == 'ln':
        self.n2 = nn.GroupNorm(1, 16)
        # for now, we hard coded this network
        # i.e. we fix the number of hidden layers i.e. 2 layers
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)
    
    def f(self, x, start_layer_idx):
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.n1(x)
        # x = self.pool(F.relu(self.conv2(x)))
        # x = self.n2(x)
        # x = x.view(-1, 16 * 4 * 4)
        if start_layer_idx == -1:
            return self.fc3(x)
        
        if cfg['norm'] in ['cn', 'pccn']:
            norm1, norm2 = None, None
        else:
            norm1, norm2 = self.n1, self.n2

        # x = apply_norm(x, self.conv1, norm1)
        # x = self.n1(self.conv1(x))
        # x = self.pool(F.relu(x))
        # x = self.n2(self.conv2(x))
        # # x = apply_norm(x, self.conv2, norm2)
        # x = self.pool(F.relu(x))
        # x = x.view(-1, 16 * 4 * 4)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.n1(self.conv1(x))
        x = self.pool(F.relu(x))
        x = self.n2(self.conv2(x))
        # x = apply_norm(x, self.conv2, norm2)
        x = self.pool(F.relu(x))
        x = x.view(-1, 16 * 5 * 5)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # print('fc2 output shape', x.shape, flush=True)
        # x = self.fc3(x)
        return x
        
    def forward(self, input, start_layer_idx=None):
        if start_layer_idx == -1:
            output = {}
            output['target'] = self.f(input['data'], start_layer_idx)
            return output
        
        output = {}
        output['target'] = self.f(input['data'], start_layer_idx)
        output['loss'] = loss_fn(output['target'], input['target'])
        return output

class SimpleCNN(nn.Module):
    def __init__(self, input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10):
        super(SimpleCNN, self).__init__()
        # self.n1 = nn.GroupNorm(1, input_dim)
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        if cfg['norm'] == 'ln' or cfg['norm'] == 'relu-ln':
            self.n1 = nn.GroupNorm(1, 6)
        self.conv2 = nn.Conv2d(6, 16, 5)
        if cfg['norm'] == 'ln' or cfg['norm'] == 'relu-ln':
            self.n2 = nn.GroupNorm(1, 16)

        # if cfg['norm'] == 'cn':
        #     self.conv1 = Conv2dCosineNorm(3, 6, 5)
        #     self.conv2 = Conv2dCosineNorm(6, 16, 5)
        # elif cfg['norm'] == 'pccn':
        #     self.conv1 = Conv2dPCCNorm(3, 6, 5)
        #     self.conv2 = Conv2dPCCNorm(6, 16, 5)
    
        # for now, we hard coded this network
        # i.e. we fix the number of hidden layers i.e. 2 layers
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)

        # if cfg['norm'] == 'cn':
        #     self.fc1 = FFNCosineNorm(input_dim, hidden_dims[0])
        #     self.fc2 = FFNCosineNorm(hidden_dims[0], hidden_dims[1])
            # self.fc3 = FFNCosineNorm(hidden_dims[1], output_dim)
        # elif cfg['norm'] == 'pccn':
        #     self.fc1 = FFNPCCNorm(input_dim, hidden_dims[0])
        #     self.fc2 = FFNPCCNorm(hidden_dims[0], hidden_dims[1])
            # self.fc3 = FFNPCCNorm(hidden_dims[1], output_dim)

    def f(self, x, start_layer_idx):
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.n1(x)
        # # print('normalize', flush=True)
        # x = self.pool(F.relu(self.conv2(x)))
        # x = self.n2(x)
        # x = x.view(-1, 16 * 5 * 5)
        if start_layer_idx == -1:
            return self.fc3(x)

        # if cfg['norm'] in ['cn', 'pccn']:
        #     norm1, norm2 = None, None
        # else:
        #     norm1, norm2 = self.n1, self.n2

        # x = apply_norm(x, self.conv1, norm1)
        # getattr(self, 'n1', None)
        if cfg['norm'] == 'relu-ln':
            x = self.pool(F.relu(self.conv1(x)))
            x = self.n1(x)
            x = self.pool(F.relu(self.conv2(x)))
            x = self.n2(x)
        else:
            if getattr(self, 'n1', None):
                x = self.n1(self.conv1(x))
            else:
                x = self.conv1(x)
            x = self.pool(F.relu(x))
            if getattr(self, 'n2', None):
                x = self.n2(self.conv2(x))
            else:
                x = self.conv2(x)
        # x = apply_norm(x, self.conv2, norm2)
            x = self.pool(F.relu(x))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        x = self.fc3(x)
        return x
        
    def forward(self, input, start_layer_idx=None):
        if start_layer_idx == -1:
            output = {}
            output['target'] = self.f(input['data'], start_layer_idx)
            return output
        
        output = {}
        output['target'] = self.f(input['data'], start_layer_idx)
        output['loss'] = loss_fn(output['target'], input['target'])
        return output

def create_CNN():
    model = None
    target_size = cfg['target_size']
    if cfg['data_name'] in ['CIFAR10', 'CIFAR100']:
        model = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=target_size)
    elif cfg['data_name'] in ['FEMNIST', 'MNIST']:
        model = SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=target_size)
    else:
        raise ValueError('wrong dataset name')
    model.apply(init_param)
    return model

# class CNN(nn.Module):
#     def __init__(self, dataset='mnist', model='cnn'):
#         super(CNN, self).__init__()
#         # define network layers
#         print("Creating model for {}".format(dataset))
#         self.dataset = dataset
#         # hidden_dim 中间的
#         # latent_dim
#         configs, input_channel, self.output_dim, self.hidden_dim, self.latent_dim=CONFIGS_[dataset]
#         print('Network configs:', configs)
#         self.named_layers, self.layers, self.layer_names =self.build_network(
#             configs, input_channel, self.output_dim)
#         self.n_parameters = len(list(self.parameters()))
#         self.n_share_parameters = len(self.get_encoder())

#     def get_number_of_parameters(self):
#         pytorch_total_params=sum(p.numel() for p in self.parameters() if p.requires_grad)
#         return pytorch_total_params

#     def build_network(self, configs, input_channel, output_dim):
#         layers = nn.ModuleList()
#         named_layers = {}
#         layer_names = []
#         kernel_size, stride, padding = 3, 2, 1
#         for i, x in enumerate(configs):
#             if x == 'Flatten':
#                 layer_name='flatten{}'.format(i)
#                 layer=nn.Flatten(1)
#                 layers+=[layer]
#                 layer_names+=[layer_name]
#             elif x == 'MaxPooling':
#                 pool_layer = nn.MaxPool2d(kernel_size=2, stride=2)
#                 layer_name = 'pool{}'.format(i)
#                 layers += [pool_layer]
#                 layer_names += [layer_name]
#             else:
#                 cnn_name = 'encode_cnn{}'.format(i)
#                 cnn_layer = nn.Conv2d(input_channel, x, stride=stride, kernel_size=kernel_size, padding=padding)
#                 named_layers[cnn_name] = [cnn_layer.weight, cnn_layer.bias]

#                 bn_name = 'encode_batchnorm{}'.format(i)
#                 bn_layer = nn.BatchNorm2d(x)
#                 named_layers[bn_name] = [bn_layer.weight, bn_layer.bias]

#                 relu_name = 'relu{}'.format(i)
#                 relu_layer = nn.ReLU(inplace=True)# no parameters to learn

#                 layers += [cnn_layer, bn_layer, relu_layer]
#                 layer_names += [cnn_name, bn_name, relu_name]
#                 input_channel = x

#         # finally, classification layer
#         fc_layer_name1 = 'encode_fc1'
#         fc_layer1 = nn.Linear(self.hidden_dim, self.latent_dim)
#         layers += [fc_layer1]
#         layer_names += [fc_layer_name1]
#         named_layers[fc_layer_name1] = [fc_layer1.weight, fc_layer1.bias]

#         fc_layer_name = 'decode_fc2'
#         fc_layer = nn.Linear(self.latent_dim, self.output_dim)
#         layers += [fc_layer]
#         layer_names += [fc_layer_name]
#         named_layers[fc_layer_name] = [fc_layer.weight, fc_layer.bias]
#         return named_layers, layers, layer_names


#     def get_parameters_by_keyword(self, keyword='encode'):
#         params=[]
#         for name, layer in zip(self.layer_names, self.layers):
#             if keyword in name:
#                 #layer = self.layers[name]
#                 params += [layer.weight, layer.bias]
#         return params

#     def get_encoder(self):
#         return self.get_parameters_by_keyword("encode")

#     def get_decoder(self):
#         return self.get_parameters_by_keyword("decode")

#     def get_shared_parameters(self, detach=False):
#         return self.get_parameters_by_keyword("decode_fc2")

#     def get_learnable_params(self):
#         return self.get_encoder() + self.get_decoder()

#     def forward(self, x, start_layer_idx = 0, logit=False):
#         """
#         :param x:
#         :param logit: return logit vector before the last softmax layer
#         :param start_layer_idx: if 0, conduct normal forward; otherwise, forward from the last few layers (see mapping function)
#         :return:
#         """
#         x = x['input']
#         if start_layer_idx < 0: #
#             return self.mapping(x, start_layer_idx=start_layer_idx, logit=logit)
#         restults={}
#         z = x
#         for idx in range(start_layer_idx, len(self.layers)):
#             layer_name = self.layer_names[idx]
#             layer = self.layers[idx]
#             z = layer(z)

#         if self.output_dim > 1:
#             restults['output'] = F.log_softmax(z, dim=1)
#         else:
#             restults['output'] = z
#         if logit:
#             restults['logit']=z
#         return restults

#     def mapping(self, z_input, start_layer_idx=-1, logit=True):
#         z = z_input
#         n_layers = len(self.layers)
#         for layer_idx in range(n_layers + start_layer_idx, n_layers):
#             layer = self.layers[layer_idx]
#             z = layer(z)
#         if self.output_dim > 1:
#             out=F.log_softmax(z, dim=1)
#         result = {'output': out}
#         if logit:
#             result['logit'] = z
#         return result
