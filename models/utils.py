import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2dCosineNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias)

        self.temp_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False)

    def forward(self, x):        

        w = self.conv.weight.data.clone()

        x2 = torch.pow(x, 2)
        w1 = torch.ones_like(w)
        self.temp_conv.weight.data = w1
        x2_len = self.temp_conv(x2)
        x_len = torch.sqrt(x2_len).detach()

        x1 = torch.ones_like(x)
        w2 = torch.pow(w, 2)
        self.temp_conv.weight.data = w2
        w2_len = self.temp_conv(x1)
        w_len = torch.sqrt(w2_len).detach()

        out = self.conv(x)
        temp = copy.deepcopy(out.detach())
        if self.bias:
            out -= self.conv.bias.data.view(1, -1, 1, 1)
        out = torch.div(out, (x_len * w_len)).clamp(min=1e-10)  # Avoid division by zero
        max_val, max_idx = torch.max(out.view(-1), dim=0)
        if max_val > 1:
            print(max_val) 
            print(temp.view(-1)[max_idx])
            print(x_len.view(-1)[max_idx])
            print(w_len.view(-1)[max_idx])
        if self.bias:
            out += self.conv.bias.data.view(1, -1, 1, 1)
        # Flatten the tensor and find the maximum element
        
        return out
        # return out

class FFNCosineNorm(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.temp_linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x):        

        w = self.linear.weight.data.clone()

        x2 = torch.pow(x, 2)
        w1 = torch.ones_like(w)
        self.temp_linear.weight.data = w1
        x2_len = self.temp_linear(x2)
        x_len = torch.sqrt(x2_len).detach()

        x1 = torch.ones_like(x)
        w2 = torch.pow(w, 2)
        self.temp_linear.weight.data = w2
        w2_len = self.temp_linear(x1)
        w_len = torch.sqrt(w2_len).detach()

        out = self.linear(x)
        temp = copy.deepcopy(out.detach())
        if self.bias:
            out -= self.linear.bias.data.view(1, -1)
        out = torch.div(out, (x_len * w_len)).clamp(min=1e-10)  # Avoid division by zero
        max_val, max_idx = torch.max(out.view(-1), dim=0)
        if max_val > 1:
            print(max_val) 
            print(temp.view(-1)[max_idx])
            print(x_len.view(-1)[max_idx])
            print(w_len.view(-1)[max_idx])
        if self.bias:
            out += self.linear.bias.data.view(1, -1)
        # Flatten the tensor and find the maximum element
        
        return out
    
class Conv2dPCCNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False)

        self.temp_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False)

    def forward(self, x):        

        w = self.conv.weight.data.clone()

        # output_channel, input_channel, kernel_size, kernel_size 
        w_mean = w.mean(dim=[1, 2, 3], keepdim=True)
        # batch_size, input_channel, height, width
        x_mean = x.mean(dim=[1, 2, 3], keepdim=True)

        w = w - w_mean
        x = x - x_mean

        x2 = torch.pow(x, 2)
        w1 = torch.ones_like(w)
        self.temp_conv.weight.data = w1
        x2_len = self.temp_conv(x2)
        x_len = torch.sqrt(x2_len).detach()

        x1 = torch.ones_like(x)
        w2 = torch.pow(w, 2)
        self.temp_conv.weight.data = w2
        w2_len = self.temp_conv(x1)
        w_len = torch.sqrt(w2_len).detach()

        self.conv.weight.data = w.clone()
        out = self.conv(x)
        
        temp = copy.deepcopy(out.detach())
        if self.bias:
            out -= self.conv.bias.data.view(1, -1, 1, 1)
        out = torch.div(out, (x_len * w_len)).clamp(min=1e-10)  # Avoid division by zero
        max_val, max_idx = torch.max(out.view(-1), dim=0)
        if max_val > 1:
            print(max_val) 
            print(temp.view(-1)[max_idx])
            print(x_len.view(-1)[max_idx])
            print(w_len.view(-1)[max_idx])
        if self.bias:
            out += self.conv.bias.data.view(1, -1, 1, 1)
        # Flatten the tensor and find the maximum element
        
        return out
    
class FFNPCCNorm(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.temp_linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x):        

        w = self.linear.weight.data.clone()

        # out_features, in_features
        w_mean = w.mean(dim=[1], keepdim=True)
        # batch_size, in_features
        x_mean = x.mean(dim=[1], keepdim=True)

        temp = w.clone()
        w = w - w_mean
        a = temp - w
        x = x - x_mean

        x2 = torch.pow(x, 2)
        w1 = torch.ones_like(w)
        self.temp_linear.weight.data = w1
        x2_len = self.temp_linear(x2)
        x_len = torch.sqrt(x2_len).detach()

        x1 = torch.ones_like(x)
        w2 = torch.pow(w, 2)
        self.temp_linear.weight.data = w2
        w2_len = self.temp_linear(x1)
        w_len = torch.sqrt(w2_len).detach()

        self.linear.weight.data = w.clone()
        out = self.linear(x)
        
        temp = copy.deepcopy(out.detach())
        if self.bias:
            out -= self.linear.bias.data.view(1, -1)
        out = torch.div(out, (x_len * w_len)).clamp(min=1e-10)  # Avoid division by zero
        max_val, max_idx = torch.max(out.view(-1), dim=0)
        if max_val > 1:
            print(max_val) 
            print(temp.view(-1)[max_idx])
            print(x_len.view(-1)[max_idx])
            print(w_len.view(-1)[max_idx])
        if self.bias:
            out += self.linear.bias.data.view(1, -1)
        # Flatten the tensor and find the maximum element
        
        return out
# def conv2d_cosnorm(self, x, w, stride, padding, bias=0.0001):
#     # x_shape = x.shape
#     # x_b = torch.full((x_shape[0], x_shape[1], x_shape[2], 1), bias, device=x.device)
#     # x = torch.cat([x_b, x], dim=3)

#     # w_shape = w.shape
#     # w_b = torch.full((w_shape[0], w_shape[1], 1, w_shape[3]), bias, device=w.device)
#     # w = torch.cat([w_b, w], dim=2)

#     x2 = torch.pow(x, 2)
#     w1 = torch.ones_like(w)
#     x2_len = F.conv2d(x2, w1, stride=stride, padding=padding)
#     x_len = torch.sqrt(x2_len)
#     torch.autograd.detect_anomaly(x_len, "x_len error")

#     x1 = torch.ones_like(x)
#     w2 = torch.pow(w, 2)
#     w2_len = F.conv2d(x1, w2, stride=stride, padding=padding)
#     w_len = torch.sqrt(w2_len)
#     torch.autograd.detect_anomaly(w_len, "w_len error")

#     y = F.conv2d(x, w, stride=stride, padding=padding)

#     return torch.div(y, (x_len * w_len)).clamp(min=1e-10)  # Avoid division by zero



def apply_norm(x, layer, norm=None):
    if norm is not None:
        x = norm(layer(x))
    else:
        x = layer(x)
    return x


def init_param(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        # if m.bias is not None:
        m.bias.data.zero_()
        # m.bias.data.zero_()
    return m


def make_batchnorm(m, momentum, track_running_stats):
    if isinstance(m, nn.BatchNorm2d):
        m.momentum = momentum
        m.track_running_stats = track_running_stats
        if track_running_stats:
            m.register_buffer('running_mean', torch.zeros(m.num_features, device=m.weight.device))
            m.register_buffer('running_var', torch.ones(m.num_features, device=m.weight.device))
            m.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long, device=m.weight.device))
        else:
            m.running_mean = None
            m.running_var = None
            m.num_batches_tracked = None
    return m


def loss_fn(output, target, reduction='mean'):
    if target.dtype == torch.int64:
        # print(f'cross_entropy: {output}, {target}')
        loss = F.cross_entropy(output, target, reduction=reduction)
        # if torch.isnan(loss) or loss.item() > 20:
        #     print(f'cross_entropy: {loss}')
        #     print(f'output: {output}\n')
        #     print(f'target: {target}\n')
        #     a = 5
    else:
        # print(f'mse: {output}, {target}')
        raise ValueError('not cross_entropy')
        loss = F.mse_loss(output, target, reduction=reduction)
        # print(f'mse_loss: {loss}')
    return loss


def mse_loss(output, target, weight=None):
    mse = F.mse_loss(output, target, reduction='none')
    mse = weight * mse if weight is not None else mse
    mse = torch.sum(mse)
    mse /= output.size(0)
    return mse


def cross_entropy_loss(output, target, weight=None):
    target = (target.topk(1, 1, True, True)[1]).view(-1)
    ce = F.cross_entropy(output, target, reduction='mean', weight=weight)
    return ce


def kld_loss(output, target, weight=None, T=1):
    kld = F.kl_div(F.log_softmax(output, dim=-1), F.softmax(target / T, dim=-1), reduction='none')
    kld = weight * kld if weight is not None else kld
    kld = torch.sum(kld)
    kld /= output.size(0)
    return kld






