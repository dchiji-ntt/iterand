
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GetSubnet(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, sparsity, zeros, ones):
        k_val = percentile(scores, sparsity*100)
        out = torch.where(scores < k_val, zeros.to(scores.device), ones.to(scores.device))
        return out

    @staticmethod
    def backward(ctx, g):
        return g, None, None, None

def percentile(t, q):
    k = 1 + round(.01 * float(q) * (t.numel() - 1))
    return t.view(-1).kthvalue(k).values.item()


class SparseModule(nn.Module):
    def init_param_(self, param, init_mode=None, scale=None):
        if init_mode == 'kaiming_normal':
            nn.init.kaiming_normal_(param, mode="fan_in", nonlinearity="relu")
            param.data *= scale
        elif init_mode == 'uniform(-1,1)':
            nn.init.uniform_(param, a=-1, b=1)
            param.data *= scale
        elif init_mode == 'kaiming_uniform':
            nn.init.kaiming_uniform_(param, mode='fan_in', nonlinearity='relu')
            param.data *= scale
        elif init_mode == 'signed_constant':
            # From github.com/allenai/hidden-networks
            fan = nn.init._calculate_correct_fan(param, 'fan_in')
            gain = nn.init.calculate_gain('relu')
            std = gain / math.sqrt(fan)
            nn.init.kaiming_normal_(param)    # use only its sign
            param.data = param.data.sign() * std
            param.data *= scale
        else:
            raise NotImplementedError

    def rerandomize_(self, param, mask, mode=None, la=None, mu=None,
                     init_mode=None, scale=None, param_twin=None):
        if param_twin is None:
            raise NotImplementedError
        else:
            param_twin = param_twin.to(param.device)

        with torch.no_grad():
            if mode == 'bernoulli':
                assert (la is not None) and (mu is None)
                rnd = param_twin
                self.init_param_(rnd, init_mode=init_mode, scale=scale)
                ones = torch.ones(param.size()).to(param.device)
                b = torch.bernoulli(ones * la)

                t1 = param.data * mask
                t2 = param.data * (1 - mask) * (1 - b)
                t3 = rnd.data * (1 - mask) * b

                param.data = t1 + t2 + t3
            elif mode == 'manual':
                assert (la is not None) and (mu is not None)

                t1 = param.data * (1 - mask)
                t2 = param.data * mask

                rnd = param_twin
                self.init_param_(rnd, init_mode=init_mode, scale=scale)
                rnd *= (1 - mask)

                param.data = (t1*la + rnd.data*mu) + t2
            else:
                raise NotImplementedError


class SparseConv2d(SparseModule):
    def __init__(self, in_ch, out_ch, **kwargs):
        super().__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch

        self.kernel_size = kwargs['kernel_size']
        self.stride = kwargs['stride'] if 'stride' in kwargs else 1
        self.padding = kwargs['padding'] if 'padding' in kwargs else 0
        self.bias_flag = kwargs['bias'] if 'bias' in kwargs else True
        self.padding_mode = kwargs['padding_mode'] if 'padding_mode' in kwargs else None

        cfg = kwargs['cfg']
        self.sparsity = cfg['conv_sparsity']
        self.init_mode = cfg['init_mode']
        self.init_mode_mask = cfg['init_mode_mask']
        self.init_scale = cfg['init_scale']
        self.init_scale_score = cfg['init_scale_score']
        self.rerand_rate = cfg['rerand_rate']
        self.function = F.conv2d

        self.initialize_weights(2)

    def initialize_weights(self, convdim=None):
        if convdim == 1:
            self.weight = nn.Parameter(torch.ones(self.out_ch, self.in_ch, self.kernel_size))
        elif convdim == 2:
            self.weight = nn.Parameter(torch.ones(self.out_ch, self.in_ch, self.kernel_size, self.kernel_size))
        else:
            raise NotImplementedError

        self.weight_score = nn.Parameter(torch.ones(self.weight.size()))
        self.weight_score.is_score = True
        self.weight_score.sparsity = self.sparsity

        self.weight_twin = torch.zeros(self.weight.size())
        self.weight_twin.requires_grad = False

        if self.bias_flag:
            self.bias = nn.Parameter(torch.zeros(self.out_ch))
        else:
            self.bias = None

        self.init_param_(self.weight_score, init_mode=self.init_mode_mask, scale=self.init_scale_score)
        self.init_param_(self.weight, init_mode=self.init_mode, scale=self.init_scale)

        self.weight_zeros = torch.zeros(self.weight_score.size())
        self.weight_ones = torch.ones(self.weight_score.size())
        self.weight_zeros.requires_grad = False
        self.weight_ones.requires_grad = False

    def get_subnet(self, weight_score=None):
        if weight_score is None:
            weight_score = self.weight_score

        subnet = GetSubnet.apply(self.weight_score, self.sparsity,
                                 self.weight_zeros, self.weight_ones)
        return subnet

    def forward(self, input):
        subnet = self.get_subnet(self.weight_score)
        pruned_weight = self.weight * subnet
        ret = self.function(
            input, pruned_weight, self.bias, self.stride, self.padding,
        )
        return ret

    def rerandomize(self, mode, la, mu):
        rate = self.rerand_rate
        mask = GetSubnet.apply(self.weight_score, self.sparsity * rate,
                               self.weight_zeros, self.weight_ones)
        scale = self.init_scale
        self.rerandomize_(self.weight, mask, mode, la, mu,
                self.init_mode, scale, self.weight_twin)

class SparseConv1d(SparseConv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.function = F.conv1d
        self.initialize_weights(1)


class SparseLinear(SparseModule):
    def __init__(self, in_ch, out_ch, bias=True, cfg=None):
        super().__init__()

        if cfg['linear_sparsity'] is not None:
            self.sparsity = cfg['linear_sparsity']
        else:
            self.sparsity = cfg['conv_sparsity']

        if cfg['init_mode_linear'] is not None:
            self.init_mode = cfg['init_mode_linear']
        else:
            self.init_mode = cfg['init_mode']

        self.init_mode_mask = cfg['init_mode_mask']
        self.init_scale = cfg['init_scale']
        self.init_scale_score = cfg['init_scale_score']
        self.rerand_rate = cfg['rerand_rate']

        self.weight = nn.Parameter(torch.ones(out_ch, in_ch))
        self.weight_score = nn.Parameter(torch.ones(self.weight.size()))
        self.weight_score.is_score = True
        self.weight_score.sparsity = self.sparsity
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_ch))
        else:
            self.bias = None

        self.weight_twin = torch.zeros(self.weight.size())
        self.weight_twin.requires_grad = False

        self.init_param_(self.weight_score, init_mode=self.init_mode_mask, scale=self.init_scale_score)
        self.init_param_(self.weight, init_mode=self.init_mode, scale=self.init_scale)

        self.weight_zeros = torch.zeros(self.weight_score.size())
        self.weight_ones = torch.ones(self.weight_score.size())
        self.weight_zeros.requires_grad = False
        self.weight_ones.requires_grad = False

    def forward(self, x, manual_mask=None):
        if manual_mask is None:
            subnet = GetSubnet.apply(self.weight_score, self.sparsity,
                                     self.weight_zeros, self.weight_ones)
            pruned_weight = self.weight * subnet
        else:
            pruned_weight = self.weight * manual_mask

        ret = F.linear(x, pruned_weight, self.bias)
        return ret

    def rerandomize(self, mode, la, mu, manual_mask=None):
        if manual_mask is None:
            rate = self.rerand_rate
            mask = GetSubnet.apply(self.weight_score, self.sparsity * rate,
                                   self.weight_zeros, self.weight_ones)
        else:
            mask = manual_mask

        scale = self.init_scale
        self.rerandomize_(self.weight, mask, mode, la, mu,
                self.init_mode, scale, self.weight_twin)

