
import cv2
import os
import numpy as np
from maskrcnn_benchmark.utils.miscellaneous import mkdir
import tifffile as tiff
from inference.metrics import mask2out, removeoverlap
from maskrcnn_benchmark.config import cfg
from inference.cell_predictor import CellDemo
from maskrcnn_benchmark.modeling.detector import build_detection_model

def infer_fluo2tcga(wts_root, out_pred_root):

    config_file = "../configs/uda_nuclei_seg/e2e_mask_rcnn_R_101_FPN_1x_gn.yaml"

    # update the config options with the config file
    cfg.merge_from_file(config_file)
    # manual override some options
    cfg.merge_from_list(["MODEL.DEVICE", "cuda"])
    model = build_detection_model(cfg)

    cell_demo = CellDemo(
        cfg,
        min_image_size=1000,
        confidence_threshold=0.5,
        weight= wts_root,
        model=model
    )

    # put your testing images
    test_root_name = ''


    # output saving root
    mkdir(out_pred_root)

    test_imgs = os.listdir(test_root_name)
    for img_name in test_imgs:

        if img_name.endswith(".png"):
            image = cv2.imread(os.path.join(test_root_name, img_name))

            predictions, mask_list = cell_demo.run_on_opencv_image(image)

            masks_no_overlap, bi_map, num_mask = removeoverlap(mask_list)

            out_name = os.path.join(out_pred_root, img_name.split('.')[0] + '.tif')
            pred_ins = mask2out(masks_no_overlap, num_mask)

            cv2.imwrite(os.path.join(out_pred_root, img_name), predictions)
            cv2.imwrite(os.path.join(out_pred_root, 'bi_mask_' + img_name), (bi_map * 255).astype(np.uint8))
            tiff.imsave(out_name, pred_ins)


if __name__ == "__main__":
    wts_root = ''
    out_pred_root = ''

    infer_fluo2tcga(wts_root, out_pred_root)
