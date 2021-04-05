import torch
import numpy as np
from torch.autograd import Variable


def build_source_label(cfg):
    need_backprop = torch.FloatTensor(1)
    device = torch.device(cfg.MODEL.DEVICE)
    if device is 'cuda':
        need_backprop = need_backprop.cuda()
    need_backprop = Variable(need_backprop)
    np_need_backprop = np.ones((1,),dtype=np.float32)
    need_backprop.data.resize_(np_need_backprop.size()).copy_(np_need_backprop)

    return need_backprop


def build_target_label(cfg):
    need_backprop = torch.FloatTensor(1)
    device = torch.device(cfg.MODEL.DEVICE)
    if device is 'cuda':
        need_backprop = need_backprop.cuda()
    need_backprop = Variable(need_backprop)
    np_need_backprop = np.zeros((1,), dtype=np.float32)
    need_backprop.data.resize_(np_need_backprop.size()).copy_(np_need_backprop)

    return need_backprop