#!/usr/bin/env bash



# fluo2tnbc

#
#CUDA_VISIBLE_DEVICES=0 python tools/train_net.py \
#    --config-file "configs/uda_nuclei_seg/e2e_mask_rcnn_R_101_FPN_1x_gn_fluo2tnbc.yaml" \
#    OUTPUT_DIR ./fluo2tnbc-models/pdam

# fluo2tcga

CUDA_VISIBLE_DEVICES=0 python tools/train_net.py \
    --config-file "./configs/uda_nuclei_seg/e2e_mask_rcnn_R_101_FPN_1x_gn_fluo2tcga.yaml" \
    OUTPUT_DIR ./fluo2tcga-models/pdam


# em epfl2vnc
#
#CUDA_VISIBLE_DEVICES=0 python tools/train_net.py \
#    --config-file "configs/uda_nuclei_seg/e2e_mask_rcnn_R_101_FPN_1x_gn_epfl2vnc.yaml" \
#    OUTPUT_DIR ./epfl2vnc-models/pdam