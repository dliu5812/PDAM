from inference.colormap import colormap
import cv2
import numpy as np
import os
import tifffile as tiff


def mkdir(out_folder):
    try:
        os.stat(os.path.dirname(out_folder + '/'))
    except:
        os.mkdir(os.path.dirname(out_folder + '/'))

def color_instance(img_folder, ins_folder, out_folder):

    mkdir(out_folder)
    imglist = os.listdir(img_folder)

    for imgname in imglist:

        imgpath = os.path.join(img_folder, imgname)
        inspath = os.path.join(ins_folder, imgname.split('.')[0] + '.tif')

        ins_seg = tiff.imread(inspath)
        img = cv2.imread(imgpath)

        masknum = np.amax(ins_seg)
        #print(masknum)

        color_list = colormap()

        for idx in range(1, masknum + 1):
            color_mask = color_list[idx % len(color_list), 0:3]

            # bi_mask_map = (ins_seg == idx).astype(np.uint8)
            bi_mask_map = (ins_seg == idx)

            ins = np.nonzero(bi_mask_map)

            img = img.astype(np.float64)
            img[ins[0], ins[1], :] *= 0.3
            img[ins[0], ins[1], :] += 0.7 * color_mask

        out_name = imgname
        cv2.imwrite(os.path.join(out_folder, out_name), img.astype(np.uint8))



if __name__ == "__main__":

    img_folder = ''
    ins_folder = ''
    out_folder = ''

    color_instance(img_folder, ins_folder, out_folder)

