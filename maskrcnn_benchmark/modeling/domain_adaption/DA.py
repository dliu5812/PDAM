

import torch.nn as nn
from torch.autograd import Function
from maskrcnn_benchmark.modeling.make_layers import conv_with_kaiming_uniform
from maskrcnn_benchmark.modeling.make_layers import make_fc
import torch.nn.functional as F
import torch


class GRLayerF(Function):

    @staticmethod
    def forward(ctx, input, grl_alpha):
        ctx.alpha= grl_alpha
        # print('grl alpha in cfg is, ', grl_alpha)

        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_outputs):
        output=grad_outputs.neg() * ctx.alpha
        return output, None

def grad_reverse(x, grl_alpha = 1.0):
    return GRLayerF.apply(x, grl_alpha)



class ResBlockDA(nn.Module):
    def __init__(self, dim, kernel_size = 3, use_gn = False):
        super(ResBlockDA, self).__init__()

        self.dim = dim
        self.Conv1 = conv_with_kaiming_uniform(use_gn=use_gn)(self.dim, self.dim, kernel_size, 1)
        self.Conv2 = conv_with_kaiming_uniform(use_gn=use_gn)(self.dim, self.dim, kernel_size, 1)

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        res = x
        x = self.relu(self.Conv1(x))
        x = self.Conv2(x)

        out = self.relu(res + x)

        return out


# down sampling fpn features for image level da, input： 8 x 8 x 256

class _ImageDA(nn.Module):
    def __init__(self, cfg, feat_out=32):
        super(_ImageDA, self).__init__()
        self.dim= cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS   # feat layer          256*H*W for vgg16
        use_gn_sem = cfg.MODEL.DOMAIN_ADAPTION.USE_GN_SEM
        self.Conv1 = conv_with_kaiming_uniform(use_gn=use_gn_sem)(256, 256, 3, 1)
        self.Conv2 = conv_with_kaiming_uniform(use_gn=use_gn_sem)(256, 512, 3, 1)
        self.Conv3 = conv_with_kaiming_uniform(use_gn=use_gn_sem)(512, 512, 3, 1)
        self.Conv_final = conv_with_kaiming_uniform()(512, 2, 1)
        self.Conv_feat = conv_with_kaiming_uniform(use_gn=use_gn_sem)(512, feat_out, 3, 1)
        # self.reLu=nn.LeakyReLU(inplace=False)
        self.reLu = nn.ReLU(inplace=False)


    def forward(self,x):
        # x = GRLayerF.apply(x, grl_alpha)
        x = self.reLu(self.Conv1(x))
        x = self.reLu(self.Conv2(x))
        x = self.reLu(self.Conv3(x))
        domain_pred = self.Conv_final(x)
        domain_feat = self.reLu(self.Conv_feat(x))

        return domain_pred, domain_feat




# Deep cnn version, with 6 layers
class _SemsegDA(nn.Module):
    def __init__(self, cfg,feat_out=32):
        super(_SemsegDA, self).__init__()
        use_gn_sem = cfg.MODEL.DOMAIN_ADAPTION.USE_GN_SEM

        self.Conv1 = conv_with_kaiming_uniform(use_gn=use_gn_sem)(128, 256, 3, 2)
        self.Conv2 = conv_with_kaiming_uniform(use_gn=use_gn_sem)(256, 512, 3, 2)
        self.Conv3 = conv_with_kaiming_uniform(use_gn=use_gn_sem)(512, 512, 3, 1)
        self.Conv_final = conv_with_kaiming_uniform()(512, 2, 1)
        self.Conv_feat = conv_with_kaiming_uniform(use_gn=use_gn_sem)(512, feat_out, 3)  # 1
        self.reLu = nn.ReLU(inplace=False)

    def forward(self,x):

        x = self.reLu(self.Conv1(x))
        x = self.reLu(self.Conv2(x))
        x = self.reLu(self.Conv3(x))
        domain_pred = self.Conv_final(x)
        domain_feat = self.reLu(self.Conv_feat(x))


        return domain_pred, domain_feat


# Deep cnn version, with more layers
class _SemsegDARes(nn.Module):
    def __init__(self, cfg, feat_out = 32):
        super(_SemsegDARes, self).__init__()
        use_gn_sem = cfg.MODEL.DOMAIN_ADAPTION.USE_GN_SEM
        self.Conv1 = conv_with_kaiming_uniform(use_gn=use_gn_sem)(2, 64, 7, 2)  # 7
        self.Resb1 = ResBlockDA(dim=64, kernel_size=3, use_gn= use_gn_sem)

        self.Conv2 = conv_with_kaiming_uniform(use_gn=use_gn_sem)(64, 128, 5, 2) # 5
        self.Resb2 = ResBlockDA(dim=128, kernel_size=3, use_gn= use_gn_sem)

        self.Conv3 = conv_with_kaiming_uniform(use_gn=use_gn_sem)(128, 256, 5, 2) # 5
        self.Resb3 = ResBlockDA(dim=256, kernel_size=3, use_gn= use_gn_sem)

        self.Conv4 = conv_with_kaiming_uniform(use_gn=use_gn_sem)(256, 512, 5, 2) # 5
        self.Resb4 = ResBlockDA(dim=512, kernel_size=3, use_gn= use_gn_sem)

        self.Conv_final = conv_with_kaiming_uniform()(512, 2, 1)  # 1
        self.Conv_feat = conv_with_kaiming_uniform(use_gn=use_gn_sem)(512, feat_out, 3)  # 1

        self.reLu = nn.ReLU(inplace=False)

    def forward(self,x):
        x = self.Resb1(self.reLu(self.Conv1(x)))
        x = self.Resb2(self.reLu(self.Conv2(x)))
        x = self.Resb3(self.reLu(self.Conv3(x)))
        x = self.Resb4(self.reLu(self.Conv4(x)))
        domain_pred = self.Conv_final(x)
        domain_feat = self.reLu(self.Conv_feat(x))

        return domain_pred, domain_feat



class _InstanceDA(nn.Module):
    def __init__(self, in_cn = 1024):
        super(_InstanceDA,self).__init__()

        self.dc_ip1 = make_fc(in_cn, 1024)
        self.dc_relu1 = nn.ReLU()
        self.dc_drop1 = nn.Dropout(p=0.5)

        self.dc_ip2 = make_fc(1024, 256)
        self.dc_relu2 = nn.ReLU()
        self.dc_drop2 = nn.Dropout(p=0.5)

        self.clssifer=make_fc(256,1)

    def forward(self,x):
        x=self.dc_drop1(self.dc_relu1(self.dc_ip1(x)))
        x=self.dc_drop2(self.dc_relu2(self.dc_ip2(x)))
        x = torch.sigmoid(self.clssifer(x))
        return x



class InsMiEstimatorConv(torch.nn.Module):
    r"""
    the local discriminator with architecture described in
    Figure 4 and Table 6 in appendix 1A of https://arxiv.org/pdf/1808.06670.pdf.
    input is the concatenate of
    "replicated feature vector E (with M_shape now)" + "M"

    replicated means that all pixels are the same, they are just copies.
    """
    def __init__(self, in_channels, interm_channels=512):
        super().__init__()

        # in_channels = E_size + M_channels
        self.c0 = conv_with_kaiming_uniform()(in_channels, interm_channels, 1)
        self.c1 = conv_with_kaiming_uniform()(interm_channels, interm_channels, 1)
        self.c2 = conv_with_kaiming_uniform()(interm_channels, 1, 1)

    def forward(self, x):

        score = F.relu(self.c0(x))
        score = F.relu(self.c1(score))
        score = self.c2(score)

        return score


class InsMiEstimatorFC(torch.nn.Module):
    r"""
    input of GlobalDiscriminator is the `M` in Encoder.forward, so with
    channels : num_feature * 2, in_channels
    shape    : (input_shape[0]-3*2, input_shape[1]-3*2), M_shape
    """
    def __init__(self, in_size, interm_size=512):
        super().__init__()


        self.l0 = make_fc(in_size, interm_size)
        self.l1 = make_fc(interm_size, interm_size)
        self.l2 = make_fc(interm_size, 1)

    def forward(self, x):

        out = F.relu(self.l0(x))
        out = F.relu(self.l1(out))
        out = self.l2(out)

        # see appendix 1A of https://arxiv.org/pdf/1808.06670.pdf
        # output of Table 5
        return out