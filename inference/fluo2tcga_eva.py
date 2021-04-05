
import os
import numpy as np
import tifffile as tiff
from inference.metrics import agg_jc_index, pixel_f1, remap_label, get_fast_pq
import xlwt


def evaluate_fluo2tcga(pred_root, gt_root):

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

    organ_list = ['breast', 'kidney', 'liver', 'prostate', 'bladder', 'colon', 'stomach']

    #for img_name in img_names:

    for organ in organ_list:
        for i in range (1,3):
            gt_name = organ + '_' + str(i) + '.tif'
            img_name = organ + '_' + str(i) + '.tif'
            pred_ins = tiff.imread(os.path.join(pred_root, img_name))
            gt_ins = tiff.imread(os.path.join(gt_root, gt_name))

            # compute predictions

            gt = remap_label(gt_ins, by_size=False)
            pred = remap_label(pred_ins, by_size=False)

            # object level pq
            pq_info_cur = get_fast_pq(gt, pred, match_iou=0.5)[0]
            pq_cur = pq_info_cur[2]
            pq_list.append(pq_cur)

            # object-level aji
            aji_cur = agg_jc_index(gt_ins, pred_ins)
            aji_list.append(aji_cur)

            # pixel level dice/f1 score
            f1_cur = pixel_f1(gt_ins, pred_ins)
            f1_list.append(f1_cur)

            counter = counter + 1

            ws.write(counter, 0, img_name)
            ws.write(counter, 1, aji_cur)
            ws.write(counter, 2, f1_cur)
            ws.write(counter, 3, pq_cur)

            print('The evaluation for current image:', img_name, 'is: aji score: ', aji_cur, 'f1 score: ', f1_cur)

    wb.save(pred_root + '.xls')

    aji_array = np.asarray(aji_list, dtype= np.float16)
    f1_array = np.asarray(f1_list, dtype= np.float16)
    pq_array = np.asarray(pq_list, dtype= np.float16)

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

    evaluate_fluo2tcga(pred_root, gt_root)


