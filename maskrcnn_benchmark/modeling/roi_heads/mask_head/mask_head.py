# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.structures.bounding_box import BoxList

from .roi_mask_feature_extractors import make_roi_mask_feature_extractor
from .roi_mask_predictors import make_roi_mask_predictor
from .inference import make_roi_mask_post_processor
from .loss import make_roi_mask_loss_evaluator


def keep_only_positive_boxes(boxes):
    """
    Given a set of BoxList containing the `labels` field,
    return a set of BoxList for which `labels > 0`.

    Arguments:
        boxes (list of BoxList)
    """
    assert isinstance(boxes, (list, tuple))
    assert isinstance(boxes[0], BoxList)
    assert boxes[0].has_field("labels")
    positive_boxes = []
    positive_inds = []
    num_boxes = 0
    for boxes_per_image in boxes:
        labels = boxes_per_image.get_field("labels")
        inds_mask = labels > 0
        inds = inds_mask.nonzero().squeeze(1)
        positive_boxes.append(boxes_per_image[inds])
        positive_inds.append(inds_mask)
    return positive_boxes, positive_inds


class ROIMaskHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super(ROIMaskHead, self).__init__()
        self.cfg = cfg.clone()
        self.feature_extractor = make_roi_mask_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_mask_predictor(
            cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_mask_post_processor(cfg)
        self.loss_evaluator = make_roi_mask_loss_evaluator(cfg)

    def forward(self, features, proposals, targets=None, box_logits = None, is_source = True):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the original proposals
                are returned. During testing, the predicted boxlists are returned
                with the `mask` field set
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """
        # print(proposals)

        if self.training and is_source:
            # during training, only focus on positive boxes
            all_proposals = proposals
            proposals, positive_inds = keep_only_positive_boxes(proposals)
            positive_bbox_idx = positive_inds[0].nonzero().squeeze(1)
            box_logits_positive = box_logits[positive_bbox_idx]
        elif self.training and not is_source:
            all_proposals = proposals
            box_logits_positive = box_logits

        if self.training and self.cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            x = features
            x = x[torch.cat(positive_inds, dim=0)]
        else:
            # print('proposals', proposals)
            x, roi_features = self.feature_extractor(features, proposals)


        mask_logits_pred = x

        mask_logits_da = F.adaptive_avg_pool2d(mask_logits_pred, (2, 2))

        mask_pred = self.predictor(mask_logits_pred)


        if not self.training:
            result = self.post_processor(mask_pred, proposals)
            return x, result, {}, {}, {}


        if is_source:
            loss_mask = self.loss_evaluator(proposals, mask_pred, targets)
        else:
            loss_mask = {}

        # print('mask da size,', mask_logits_da.size())
        # print('box da size,', box_logits_positive.size())


        batch_size = mask_pred.size()[0]
        mask_logits_da = mask_logits_da.view(batch_size, -1)
        ins_logits_da = mask_logits_da + box_logits_positive

        if not is_source:
            return x, all_proposals, {}, ins_logits_da, roi_features

        return x, all_proposals, dict(loss_mask=loss_mask), ins_logits_da, roi_features


def build_roi_mask_head(cfg, in_channels):
    return ROIMaskHead(cfg, in_channels)
