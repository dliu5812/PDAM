# PDAM: Unsupervised Domain Adaptive Instance Segmentation in Microscopy Images


In this project, we proposed a Panoptic Domain Adaptive Mask R-CNN (PDAM) for unsupervised instance segmentation in microscopy images.



The implementations are for our previous two papers:

[Unsupervised Instance Segmentation in Microscopy Images via Panoptic Domain Adaptation and Task Re-Weighting](https://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_Unsupervised_Instance_Segmentation_in_Microscopy_Images_via_Panoptic_Domain_Adaptation_CVPR_2020_paper.pdf), CVPR 2020.
 
[PDAM: A Panoptic-Level Feature Alignment Framework for Unsupervised Domain Adaptive Instance Segmentation in Microscopy Images](https://ieeexplore.ieee.org/abstract/document/9195030), IEEE Transactions on Medical Imaging.



## Introduction and Installation

Please follow [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) to set up the environment. In this project, the Pytorch Version 1.4.0 and CUDA 10.1 are used.


## Data

### Data Introduction

In this work, we use four datasets:

Histopathology Images: TCGA-Kumar, and TNBC. Please download them from [link](https://drive.google.com/drive/folders/1l55cv3DuY-f7-JotDN7N5nbNnjbLWchK).

For the testing images in TCGA-Kumar dataset, we rename them in the inference and evaluation process. Please refer to [Link](https://cloudstor.aarnet.edu.au/plus/s/Tpd3d6H2XxUlkl4) for details.

Fluorescence Microscopy Images: BBC039 Version 1. Download from this [link](https://bbbc.broadinstitute.org/BBBC039).

Electron Microscopy Images: [EPFL](https://www.epfl.ch/labs/cvlab/data/data-em/), and [VNC](https://github.com/unidesigner/groundtruth-drosophila-vnc).

**If you are using these datasets in your research, please also remember to cite their original work.**

### Data preparation

All the data should be put in `./dataset`. For the detailed path of each dataset, please refer to:

`./maskrcnn_benchmark/config/path_catalog.py`

Here we provide some sample images on adaptation from BBBC039V1 to TCGA-Kumar (fluo2tcga).

Note that the instance annotations are stored in .json files following the MSCOCO format. If you want to generate the annotations by yourself, please follow this [repository](https://github.com/waspinator/pycococreator).

## Model training

First, follow our paper to generate synthesized patches using [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).

Next, implement the nuclei inpainting mechanism by running `python auxiliary_nuclei_inpaint.py`. We have several demo results in `./nuclei_inpaint`.


For training the model on three UDA settings in our papers, please refer to:

`./train_gn_pdam.sh`.

## Model inference and Evaluation

The code for this part is in `./inference`. Just list the settings from BBBC039V1 to TCGA-Kumar as an example:

To get the instance segmentation prediction, run `python fluo2tcga_infer.py`. Remember to manually set the path of the pre-trained weights, testing images, and output folder.

To evaluate the segmentation performance under AJI, pixel-f1, and Panoptic Quality (PQ), please run `python fluo2tcga_eva.py`. The overall results for all the testing images will be saved in a .xls file.



## Citations (Bibtex)
Please consider citing our papers in your publications if they are helpful to your research:
```
@inproceedings{liu2020unsupervised,
  title={Unsupervised instance segmentation in microscopy images via panoptic domain adaptation and task re-weighting},
  author={Liu, Dongnan and Zhang, Donghao and Song, Yang and Zhang, Fan and O'Donnell, Lauren and Huang, Heng and Chen, Mei and Cai, Weidong},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={4243--4252},
  year={2020}
}

```

```
@article{liu2020pdam,
  title={PDAM: A Panoptic-level Feature Alignment Framework for Unsupervised Domain Adaptive Instance Segmentation in Microscopy Images},
  author={Liu, Dongnan and Zhang, Donghao and Song, Yang and Zhang, Fan and O’Donnell, Lauren and Huang, Heng and Chen, Mei and Cai, Weidong},
  journal={IEEE Transactions on Medical Imaging},
  year={2020},
  publisher={IEEE}
}


```

 
## Thanks to the Third Party Repositories

[maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)

[pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

[quip_cnn_segmentation](https://github.com/SBU-BMI/quip_cnn_segmentation)

[hover_net](https://github.com/vqdang/hover_net)



## Contact

Please contact Dongnan Liu (dongnanliu0201@gmail.com) for any questions.


## License

PDAM is released under the MIT license. See [LICENSE](LICENSE) for additional details.

