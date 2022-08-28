# Map3D-Registration: An End-to-end Pipeline for 3D Slide-wise Multi-stain Renal Pathology Registration

### [[Accelerated Pipeline Docker]](https://github.com/MASILab/SLANTbrainSeg/tree/master/python)[[Project Page]](https://github.com/hrlblab/Map3D)[[Journal Paper]](https://arxiv.org/pdf/2006.06038.pdf)[[SPIE 2023 Paper]](https://github.com/hrlblab/Map3D)<br />


This is the official implementation of Map3D-Registration: An End-to-end Pipeline for 3D Slide-wise Multi-stain Renal Pathology Registration


![Overview](https://github.com/hrlblab/Map3D/blob/main/Figure/pipeline.png)<br />

**Journal Paper** <br />
> [Map3D: Registration Based Multi-Object Tracking
on 3D Serial Whole Slide Images](https://arxiv.org/pdf/2006.06038.pdf) <br />
> Ruining Deng, Haichun Yang, Aadarsh Jha, Yuzhe Lu, Peng Chu, Agnes B. Fogo, and  Yuankai Huo. <br />


**SPIE Paper** <br />
> [An End-to-end Pipeline for 3D Slide-wise Multi-stain Renal
Pathology Registration](https://github.com/hrlblab/Map3D)<br />
> Peize Li*, Ruining Deng*, and Yuankai Huo.<br />
> *(Under review)* <br />

```diff
+ We release the registration pipeline as a single Docker.
```

## Abstract
Tissue examination and quantification in a 3D context on serial section whole slide images (WSIs) were labor-
intensive and time-consuming tasks. Our previous study proposed a novel registration-based method (Map3D)
to automatically align WSIs to the same physical space, reducing the human efforts of screening serial sections
from WSIs. However, the registration performance of our Map3D method was only evaluated on single-stain
WSIs with large-scale kidney tissue samples. In this paper, we provide a Docker for an end-to-end 3D slide-wise
registration pipeline on needle biopsy serial sections in a multi-stain paradigm. <br /> 

The contribution of this paper is three-fold: <br />
(1) We release a containerized Docker for an end-to-end multi-stain WSI registration; <br />
(2) We prove that the Map3D pipeline is capable of sectional registration from multi-stain WSI; <br />
(3) We verify that the Map3D pipeline can also be applied to needle biopsy tissue samples.

## Quick Start
#### Get our docker image

```
sudo docker pull hrlblab/MAP3D-Regis
```
#### Run Map3D-Regis
You can run the following commands to run Map3D Registration pipeline. You may change the `input_dir` and the list of indexes, and then you will have the final segmentation results in `output_dir`. Please refer to [DATA.md](https://github.com/hrlblab/Map3D/blob/main/DATA.md) for input data format requirement and data arrangement.
```
# you need to specify the input directory. 
export input_dir=/home/input_dir   

# set output directory
export output_dir=$input_dir/output

# run the docker
sudo nvidia-docker run -it --rm -v $input_dir:/INPUTS -v $output_dir:/OUTPUTS hrlblab/MAP3D

# Enter a comma seperated list of indexes to indicate which image should be used as the middle section image in each case
2,3,5
```
## Run Pipeline Locally without Docker
Please refer to [Develop.md](https://github.com/hrlblab/Map3D/blob/main/DEVELOP.md) for instructions of running Map3D Registration pipeline locally.


## Map3D Registration Demo
Omni-Seg can easily be run on a single image.

Below is an example input of region image.

<img src='GithubFigure/region_input.png' align="center" height="230px"> 

- The entire pipeline is at the [Omni_seg_pipeline_gpu](Omni_seg_pipeline_gpu/) folder
- Create three empty folders named as "40X", "10X", and "5X" under [Omni_seg_pipeline_gpu/svs_input](Omni_seg_pipeline_gpu/svs_input) folder. Put 40X, 10X and 5X PNG files of the region image into these folders correspondingly. Each folder must contain only one file when running.
- Create three empty folders in the [Omni_seg_pipeline_gpu](Omni_seg_pipeline_gpu/) folder (before running, these three folders must be empty to remove any previous data): 
  1. "clinical_patches" folder
  2. "segmentation_merge" folder
  3. "final_merge" folder
- Run the python scripts as following orders:
  1. [1024_Step1_GridPatch_overlap_padding.py](Omni_seg_pipeline_gpu/1024_Step1_GridPatch_overlap_padding.py)
  ```
  python 1024_Step1_GridPatch_overlap_padding.py
  ```
  2. [1024_Step1.5_MOTSDataset_2D_Patch_normal_save_csv.py](Omni_seg_pipeline_gpu/1024_Step1.5_MOTSDataset_2D_Patch_normal_save_csv.py)
  ```
  python 1024_Step1.5_MOTSDataset_2D_Patch_normal_save_csv.py
  ```
  3. [Random_Step2_Testing_OmniSeg_label_overlap_64_padding.py](Omni_seg_pipeline_gpu/Random_Step2_Testing_OmniSeg_label_overlap_64_padding.py)
  ```
  python Random_Step2_Testing_OmniSeg_label_overlap_64_padding.py --reload_path 'snapshots_2D/fold1_with_white_UNet2D_ns_normalwhole_1106/MOTS_DynConv_fold1_with_white_UNet2D_ns_normalwhole_1106_e89.pth'
  ```
  4. [step3.py](Omni_seg_pipeline_gpu/step3.py)
  ```
  python step3.py
  ```
  5. [step4.py](Omni_seg_pipeline_gpu/step4.py)
  ```
  python step4.py
  ```
- The output will be stored at "final_merge" folder.

If set up correctly, the output should look like

<img src='GithubFigure/region_output.png' align="center" height="230px"> 

## Omni-Seg - Whole Slide Image Demo
CircleNet can also be run on Whole Slide Images in *.svs file format.

Please download the following file:
- [Human Kidney WSI (3d90_PAS.svs)](https://vanderbilt.box.com/s/sskcgbvz15bcfuh9sra96u1dy1hzqm6o)

We need to annotate and convert data into *.png file format first.

- Annotate the WSI rectangularly to remove most of the empty background. Recommend to use ImageScope and save the .xml file for annotation information. 
- Convert the svs file into PNG files and saved into 40X, 10X and 5X magnifications. Please refer to [Omni_seg_pipeline_gpu/svs_input/svs_to_png.py](Omni_seg_pipeline_gpu/svs_input/svs_to_png.py) for an example to convert svs format to PNG format and resize to different magnifications.
- Create three empty folders named as "40X", "10X", and "5X" under [Omni_seg_pipeline_gpu/svs_input](Omni_seg_pipeline_gpu/svs_input) folder. Put 40X, 10X and 5X PNG files into these folders correspondingly. Each folder must contain only one file when running. 

After annotation, the inputs should be like the following image with three different magnifications

<img src='GithubFigure/WSI_input.png' align="center" height="350px"> 

Please create three empty folders in the [Omni_seg_pipeline_gpu](Omni_seg_pipeline_gpu/) folder (before running, these three folders must be empty to remove any previous data): 
  1. "clinical_patches" folder
  2. "segmentation_merge" folder
  3. "final_merge" folder
  
To run the Omni-Seg pipeline, please go to [Omni_seg_pipeline_gpu](Omni_seg_pipeline_gpu/) folder and run the python scipts as following orders:
  1. [1024_Step1_GridPatch_overlap_padding.py](Omni_seg_pipeline_gpu/1024_Step1_GridPatch_overlap_padding.py)
  ```
  python 1024_Step1_GridPatch_overlap_padding.py
  ```
  2. [1024_Step1.5_MOTSDataset_2D_Patch_normal_save_csv.py](Omni_seg_pipeline_gpu/1024_Step1.5_MOTSDataset_2D_Patch_normal_save_csv.py)
  ```
  python 1024_Step1.5_MOTSDataset_2D_Patch_normal_save_csv.py
  ```
  3. [Random_Step2_Testing_OmniSeg_label_overlap_64_padding.py](Omni_seg_pipeline_gpu/Random_Step2_Testing_OmniSeg_label_overlap_64_padding.py)
  ```
  python Random_Step2_Testing_OmniSeg_label_overlap_64_padding.py --reload_path 'snapshots_2D/fold1_with_white_UNet2D_ns_normalwhole_1106/MOTS_DynConv_fold1_with_white_UNet2D_ns_normalwhole_1106_e89.pth'
  ```
  4. [step3.py](Omni_seg_pipeline_gpu/step3.py)
  ```
  python step3.py
  ```
  5. [step4.py](Omni_seg_pipeline_gpu/step4.py)
  ```
  python step4.py
  ```

The output will be stored at "final_merge" folder.

If set up correctly, the output should look like

<img src='GithubFigure/WSI_output.png' align="center" height="350px"> 





## Previous Versions
#### Google Colab
A Google Colab version of the Oracle pipeline can be found [here](https://drive.google.com/drive/folders/1vKeDMYI3Xcm6s9yAy5stBhqKWjOFvUoy?usp=sharing). The code demonstrates the patch-wise segmentation of the Oracle pipeline. 

## Citation
```
@inproceedings{
deng2022omniseg,
title={Omni-Seg: A Single Dynamic Network for Multi-label Renal Pathology Image Segmentation using Partially Labeled Data},
author={Ruining Deng and Quan Liu and Can Cui and Zuhayr Asad and Haichun Yang and Yuankai Huo},
booktitle={Medical Imaging with Deep Learning},
year={2022},
url={https://openreview.net/forum?id=v-z4Zxkt9Ex}
}
```