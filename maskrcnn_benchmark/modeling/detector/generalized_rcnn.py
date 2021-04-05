# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads

from maskrcnn_benchmark.modeling.backbone.backbone import build_panoptic_fpn
from ..domain_adaption.domain_adaption_head import build_domain_adaption_head
from ..domain_adaption.domain_adaption_head import build_mi_max_head



class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)
        self.cfg = cfg

        self.pan_fpn = build_panoptic_fpn(cfg)

        self.sem_da = build_domain_adaption_head(cfg, modal='semantic')
        self.ins_da = build_domain_adaption_head(cfg, modal='instance')
        self.img_da = build_domain_adaption_head(cfg, modal='image')

        self.mi_max = build_mi_max_head(cfg)


    def forward(self, images, t_images=None, targets=None, grl_alpha = 0.1, semgt = None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        features = self.backbone(images.tensors)
        proposals, proposal_losses = self.rpn(images, features, targets)



        if self.training:
            losses = {}

            t_images = to_image_list(t_images)
            t_features = self.backbone(t_images.tensors)

            # Image level adaptation
            da_img_losses, re_weight_img, s_img_da_feat, t_img_da_feat = self.img_da(features, t_features, grl_alpha)
            losses.update(da_img_losses)

            for proposal_loss_name in proposal_losses:
                proposal_losses[proposal_loss_name] = proposal_losses[proposal_loss_name] * re_weight_img

            losses.update(proposal_losses)

            # Semantic level adaptation
            semseg_pred_s, semseg_entropy_s = self.pan_fpn(features)
            semseg_pred_t, semseg_entropy_t = self.pan_fpn(t_features)

            da_sem_losses, re_weight_sem, s_sem_da_feat, t_sem_da_feat = self.sem_da(semseg_entropy_s, semseg_entropy_t, grl_alpha)
            semseg_loss = nn.CrossEntropyLoss()(semseg_pred_s, semgt)
            losses.update(da_sem_losses)
            semseg_loss = semseg_loss * re_weight_sem
            semseg_losses = {'loss_semseg': semseg_loss}
            losses.update(semseg_losses)


            # Instance level adaptation

            x, result, detector_losses, s_ins_logits_da, s_roi_features \
                = self.roi_heads(features, proposals,targets, is_source=True)

            t_proposals, _ = self.rpn(t_images, t_features, targets=None)
            t_x, t_result, _, t_ins_logits_da_raw, t_roi_features \
                = self.roi_heads(t_features, t_proposals, targets=None, is_source=False)

            batch_ins_s = s_ins_logits_da.size()[0]
            batch_ins_t = t_ins_logits_da_raw.size()[0]


            if batch_ins_t < batch_ins_s:
                div, mod = divmod(batch_ins_s, batch_ins_t)
                t_ins_logits_da_raw = t_ins_logits_da_raw.repeat(div,1)
                t_ins_logits_da_cat = t_ins_logits_da_raw[:mod,:]
                t_ins_logits_da = torch.cat([t_ins_logits_da_raw, t_ins_logits_da_cat], dim=0)
            else:
                t_ins_logits_da = t_ins_logits_da_raw[:batch_ins_s,:]

            # t_ins_logits_da = t_ins_logits_da_raw

            da_ins_losses, re_weight_ins = self.ins_da(s_ins_logits_da, t_ins_logits_da, grl_alpha)
            losses.update(da_ins_losses)

            for det_loss_name in detector_losses:
                detector_losses[det_loss_name] = detector_losses[det_loss_name] * re_weight_ins

            losses.update(detector_losses)


            if self.cfg.MODEL.DOMAIN_ADAPTION.MI_MAX:
                da_informax_losses = self.mi_max(s_ins_logits_da, t_ins_logits_da)
                losses.update(da_informax_losses)


            return losses

        if self.roi_heads:
            x, result, detector_losses, s_ins_logits_da, s_roi_features \
                = self.roi_heads(features, proposals, targets, is_source = True)
        else:
            # RPN-only models don't have roi_heads
            result = proposals

        return result
