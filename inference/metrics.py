

import numexpr as ne

from sklearn.metrics import f1_score
import os
import numpy as np
from scipy.optimize import linear_sum_assignment

# fast version of Aggregated Jaccrd Index
def agg_jc_index(gt_ins, pred):
    from tqdm import tqdm_notebook
    """Calculate aggregated jaccard index for prediction & GT mask
    copy from: https://github.com/jimmy15923/Miccai_challenge_MONUSEG/blob/master/Aggregate_Jaccard_Index.py
    reference paper here: https://www.dropbox.com/s/j3154xgkkpkri9w/IEEE_TMI_NuceliSegmentation.pdf?dl=0
    mask: Ground truth mask, shape = [1000, 1000, instances]
    pred: Prediction mask, shape = [1000,1000], dtype = uint16, each number represent one instance
    Returns: Aggregated Jaccard index for GT & mask 
    """

    def compute_iou(m, pred, pred_mark_isused, idx_pred):
        # check the prediction has been used or not
        if pred_mark_isused[idx_pred]:
            intersect = 0
            union = np.count_nonzero(m)
        else:
            p = (pred == idx_pred)
            # replace multiply with bool operation
            s = ne.evaluate("m&p")
            intersect = np.count_nonzero(s)
            union = np.count_nonzero(m) + np.count_nonzero(p) - intersect
        return (intersect, union)

    mask = tras_gt(gt_ins)
    mask = mask.astype(np.bool)
    c = 0  # count intersection
    u = 0  # count union
    pred_instance = pred.max()  # predcition instance number
    pred_mark_used = []  # mask used
    pred_mark_isused = np.zeros((pred_instance + 1), dtype=bool)

    for idx_m in range(len(mask[0, 0, :])):
        m = np.take(mask, idx_m, axis=2)

        intersect_list, union_list = zip(
            *[compute_iou(m, pred, pred_mark_isused, idx_pred) for idx_pred in range(1, pred_instance + 1)])

        iou_list = np.array(intersect_list) / np.array(union_list)
        hit_idx = np.argmax(iou_list)
        c += intersect_list[hit_idx]
        u += union_list[hit_idx]
        pred_mark_used.append(hit_idx)
        pred_mark_isused[hit_idx + 1] = True

    pred_mark_used = [x + 1 for x in pred_mark_used]
    pred_fp = set(np.unique(pred)) - {0} - set(pred_mark_used)
    pred_fp_pixel = np.sum([np.sum(pred == i) for i in pred_fp])

    u += pred_fp_pixel
    #print(c / u)
    return (c / u)

def tras_gt(gt):
    num_ins = np.amax(gt)
    out = np.zeros([gt.shape[0], gt.shape[1], num_ins], dtype = np.uint16)# .astype(np.uint16)
    for i in range (1, num_ins + 1):
        mask_cur = (gt == i)
        out[:,:,i-1] = mask_cur
    return out


def pixel_f1(gt_ins, pred_ins):
    lbl = gt_ins
    pred = pred_ins
    lbl[lbl > 0] = 1
    pred[pred > 0] = 1
    l, p = lbl.flatten(), pred.flatten()
    pix_f1 = f1_score(l, p)

    return pix_f1


def mask2out(mask_array, num_mask):

    output = np.zeros(mask_array[:, :, 0].shape, dtype = np.uint16)  # .astype(np.uint16)

    for i in range(num_mask):
        output = output + mask_array[:, :, i] * (i + 1)

    mask_out = output.astype(np.uint16)

    return mask_out

def removeoverlap(mask_array):

    num = mask_array.shape[-1]
    bi_map = np.zeros(mask_array[:, :, 0].shape , dtype = np.uint16)# .astype(np.uint16)
    out_list = []
    for i in range(num):
        mask_cur = mask_array[:,:,i]
        overlap_cur = bi_map * mask_cur
        if mask_cur.sum() == 0:
            continue
        overlap_ratio = float(overlap_cur.sum())/ float(mask_cur.sum())
        if overlap_cur.sum() != 0 :
            if overlap_ratio > 0.7:
                continue
            mask_cur = mask_cur - overlap_cur.astype(np.uint16)
            mask_array[:, :, i] = mask_cur

        bi_map = (bi_map + mask_cur) > 0
        out_list.append(mask_cur)

    num_mask = len(out_list)
    out_array = np.array(out_list, dtype= mask_array.dtype).transpose(1, 2, 0)

    return out_array, bi_map, num_mask


def get_fast_pq(true, pred, match_iou=0.5):
    """
    `match_iou` is the IoU threshold level to determine the pairing between
    GT instances `p` and prediction instances `g`. `p` and `g` is a pair
    if IoU > `match_iou`. However, pair of `p` and `g` must be unique
    (1 prediction instance to 1 GT instance mapping).
    If `match_iou` < 0.5, Munkres assignment (solving minimum weight matching
    in bipartite graphs) is caculated to find the maximal amount of unique pairing.
    If `match_iou` >= 0.5, all IoU(p,g) > 0.5 pairing is proven to be unique and
    the number of pairs is also maximal.

    Fast computation requires instance IDs are in contiguous orderding
    i.e [1, 2, 3, 4] not [2, 3, 6, 10]. Please call `remap_label` beforehand
    and `by_size` flag has no effect on the result.
    Returns:
        [dq, sq, pq]: measurement statistic
        [paired_true, paired_pred, unpaired_true, unpaired_pred]:
                      pairing information to perform measurement

    """
    assert match_iou >= 0.0, "Cant' be negative"

    true = np.copy(true)
    pred = np.copy(pred)
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))

    true_masks = [None, ]
    for t in true_id_list[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)

    pred_masks = [None, ]
    for p in pred_id_list[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)

    # prefill with value
    pairwise_iou = np.zeros([len(true_id_list) - 1,
                             len(pred_id_list) - 1], dtype=np.float64)

    # caching pairwise iou
    for true_id in true_id_list[1:]:  # 0-th is background
        t_mask = true_masks[true_id]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        for pred_id in pred_true_overlap_id:
            if pred_id == 0:  # ignore
                continue  # overlaping background
            p_mask = pred_masks[pred_id]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            iou = inter / (total - inter)
            pairwise_iou[true_id - 1, pred_id - 1] = iou
    #
    if match_iou >= 0.5:
        paired_iou = pairwise_iou[pairwise_iou > match_iou]
        pairwise_iou[pairwise_iou <= match_iou] = 0.0
        paired_true, paired_pred = np.nonzero(pairwise_iou)
        paired_iou = pairwise_iou[paired_true, paired_pred]
        paired_true += 1  # index is instance id - 1
        paired_pred += 1  # hence return back to original
    else:  # * Exhaustive maximal unique pairing
        #### Munkres pairing with scipy library
        # the algorithm return (row indices, matched column indices)
        # if there is multiple same cost in a row, index of first occurence
        # is return, thus the unique pairing is ensure
        # inverse pair to get high IoU as minimum
        paired_true, paired_pred = linear_sum_assignment(-pairwise_iou)
        ### extract the paired cost and remove invalid pair
        paired_iou = pairwise_iou[paired_true, paired_pred]

        # now select those above threshold level
        # paired with iou = 0.0 i.e no intersection => FP or FN
        paired_true = list(paired_true[paired_iou > match_iou] + 1)
        paired_pred = list(paired_pred[paired_iou > match_iou] + 1)
        paired_iou = paired_iou[paired_iou > match_iou]

    # get the actual FP and FN
    unpaired_true = [idx for idx in true_id_list[1:] if idx not in paired_true]
    unpaired_pred = [idx for idx in pred_id_list[1:] if idx not in paired_pred]
    # print(paired_iou.shape, paired_true.shape, len(unpaired_true), len(unpaired_pred))

    #
    tp = len(paired_true)
    fp = len(unpaired_pred)
    fn = len(unpaired_true)
    # get the F1-score i.e DQ
    dq = tp / (tp + 0.5 * fp + 0.5 * fn)
    # get the SQ, no paired has 0 iou so not impact
    sq = paired_iou.sum() / (tp + 1.0e-6)

    return [dq, sq, dq * sq], [paired_true, paired_pred, unpaired_true, unpaired_pred]


def remap_label(pred, by_size=False):
    """
    Rename all instance id so that the id is contiguous i.e [0, 1, 2, 3]
    not [0, 2, 4, 6]. The ordering of instances (which one comes first)
    is preserved unless by_size=True, then the instances will be reordered
    so that bigger nucler has smaller ID
    Args:
        pred    : the 2d array contain instances where each instances is marked
                  by non-zero integer
        by_size : renaming with larger nuclei has smaller id (on-top)
    """
    pred_id = list(np.unique(pred))
    pred_id.remove(0)
    if len(pred_id) == 0:
        return pred # no label
    if by_size:
        pred_size = []
        for inst_id in pred_id:
            size = (pred == inst_id).sum()
            pred_size.append(size)
        # sort the id by size in descending order
        pair_list = zip(pred_id, pred_size)
        pair_list = sorted(pair_list, key=lambda x: x[1], reverse=True)
        pred_id, pred_size = zip(*pair_list)

    new_pred = np.zeros(pred.shape, np.int32)
    for idx, inst_id in enumerate(pred_id):
        new_pred[pred == inst_id] = idx + 1
    return new_pred