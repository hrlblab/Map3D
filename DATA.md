# Data

This document provides tutorials to develop Map3D Registration.

## New data
Basically there are three steps:

- Annotate the WSI rectangularly. Recommend to use ImageScope and save the .xml file for annotation information. 
- Convert the svs file into PNG files and saved into 10X magnifications. Put PNG files from the same series of WSI into the same folder and name these folders as "no1", "no2", and etc. For help with this step, please refer to
- Create one empty folder to be the input folder. Put "no1", "no2", ... into this input folder.


## Run Whole Slide Image segmentation pipeline locally without docker
- The entire pipeline is at the [Omni_seg_pipeline_gpu](Omni_seg_pipeline_gpu/) folder
- Create three empty folders in the [Omni_seg_pipeline_gpu](Omni_seg_pipeline_gpu/) folder (before running, these three folders must be empty to remove any previous data): 
  1. "clinical_patches" folder
  2. "segmentation_merge" folder
  3. "final_merge" folder
- Run python scripts in [Omni_seg_pipeline_gpu](Omni_seg_pipeline_gpu/) folder as following orders: 
  1. [1024_Step1_GridPatch_overlap_padding.py](Omni_seg_pipeline_gpu/1024_Step1_GridPatch_overlap_padding.py)
  2. [1024_Step1.5_MOTSDataset_2D_Patch_normal_save_csv.py](Omni_seg_pipeline_gpu/1024_Step1.5_MOTSDataset_2D_Patch_normal_save_csv.py)
  3. [Random_Step2_Testing_OmniSeg_label_overlap_64_padding.py](Omni_seg_pipeline_gpu/Random_Step2_Testing_OmniSeg_label_overlap_64_padding.py)
  4. [step3.py](Omni_seg_pipeline_gpu/step3.py)
  5. [step4.py](Omni_seg_pipeline_gpu/step4.py)
- The output will be stored at "final_merge" folder.