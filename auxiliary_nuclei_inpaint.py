import cv2
import os
import numpy as np
from PIL import Image
from skimage.color import rgb2hed
from scipy import stats




def nuclei_inpaint(img_path, lbl_path, out_path):

    try:
        os.stat(os.path.dirname(out_path + '/'))
    except:
        os.mkdir(os.path.dirname(out_path + '/'))


    img_list = os.listdir(img_path)

    for image_name in img_list:
        if not image_name.endswith('.png'):
            continue
        img_abs_path = os.path.join(img_path, image_name)
        img_rgb = np.array(Image.open(img_abs_path).convert('RGB'))
        img_hed = rgb2hed((img_rgb.astype(np.float32)/255.0).astype(np.float32))
        img_gray = (255 * (img_hed[:,:,0] - img_hed[:,:,0].min()) / (img_hed[:,:,0].max() - img_hed[:,:,0].min())).astype(np.uint8)
        nuclei_color_threshold, thresholded = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        nuclei_color_threshold_per = stats.percentileofscore(img_gray.flatten(), nuclei_color_threshold)
        nuclei_mask_pred = (img_hed[:, :, 0] > np.percentile(img_hed[:, :, 0], nuclei_color_threshold_per))


        gt_abs_path = os.path.join(lbl_path, image_name.split('.')[0] + '.png')
        gt = cv2.imread(gt_abs_path)[:,:,0]
        gt_bi = (gt > 0).astype(np.uint8)

        nuclei_mask_extra = (nuclei_mask_pred - gt_bi) == 1

        img_inpaint = cv2.inpaint(img_rgb, nuclei_mask_extra.astype(np.uint8), 3, cv2.INPAINT_TELEA)
        img_inpaint = cv2.cvtColor(img_inpaint, cv2.COLOR_BGR2RGB)

        inpaint_abs_name = os.path.join(out_path, image_name)
        cv2.imwrite(inpaint_abs_name, img_inpaint)







if __name__ == "__main__":

    img_path = 'nuclei_inpaint/raw_synthesized_patches'
    lbl_path = 'nuclei_inpaint/synthesized_labels'
    out_path = 'nuclei_inpaint/inpaint_synthesized_patches'

    nuclei_inpaint(img_path, lbl_path, out_path)