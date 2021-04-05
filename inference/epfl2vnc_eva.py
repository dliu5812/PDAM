

import os
import numpy as np
import tifffile as tiff
from inference.metrics import agg_jc_index, pixel_f1, remap_label, get_fast_pq
import xlwt


def evaluate_epfl2vnc(pred_root, gt_root):

    aji_list = []
    f1_list = []
    pq_list = []

    wb = xlwt.Workbook()
    ws = wb.add_sheet('Test Sheet')

    counter = 0

    ws.write(0, 0, 'img_name')
    ws.write(0, 1, 'aji')
    ws.write(0, 2, 'f1')
    ws.write(0, 3, 'pq')


    test_imgs = os.listdir(pred_root)


    #for img_name in img_names:

    for img_name in test_imgs:
        if img_name.endswith(".tif"):
            gt_name = img_name
            pred_ins = tiff.imread(os.path.join(pred_root, img_name))
            gt_ins = tiff.imread(os.path.join(gt_root, gt_name))

            aji_cur = agg_jc_index(gt_ins, pred_ins)
            aji_list.append(aji_cur)

            f1_cur = pixel_f1(gt_ins, pred_ins)
            f1_list.append(f1_cur)

            gt = remap_label(gt_ins, by_size=False)
            pred = remap_label(pred_ins, by_size=False)

            pq_info_cur = get_fast_pq(gt, pred, match_iou=0.5)[0]
            pq_cur = pq_info_cur[2]
            pq_list.append(pq_cur)

            counter = counter + 1

            ws.write(counter, 0, img_name)
            ws.write(counter, 1, aji_cur)
            ws.write(counter, 2, f1_cur)
            ws.write(counter, 3, pq_cur)


    wb.save(pred_root + '.xls')


    aji_array = np.asarray(aji_list, dtype= np.float32)
    f1_array = np.asarray(f1_list, dtype= np.float32)
    pq_array = np.asarray(pq_list, dtype= np.float32)


    aji_avg = np.average(aji_array)
    aji_std = np.std(aji_array)

    f1_avg = np.average(f1_array)
    f1_std = np.std(f1_array)

    pq_avg = np.average(pq_array)
    pq_std = np.std(pq_array)

    print(pred_root)

    print('average aji score of this method is: ', aji_avg, ' ', aji_std)
    print('average f1 score of this method is: ', f1_avg, ' ',f1_std)
    print('average pq score of this method is: ', pq_avg, ' ',pq_std)



if __name__ == "__main__":
    pred_root = ''
    gt_root = ''

    evaluate_epfl2vnc(pred_root, gt_root)
