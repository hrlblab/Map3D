# Develop

## Installation

We used PyTorch 1.9.1 on [Ubuntu 22.04 LTS](https://releases.ubuntu.com/22.04/) with [Anaconda](https://www.anaconda.com/download) Python 3.7.

1. [Optional but recommended] Create a new Conda environment. 

    ~~~
    conda create --name omni_seg python=3.7
    ~~~
    
    And activate the environment.
    
    ~~~
    conda activate omni_seg
    ~~~

2. Clone the Map3D repo

3. Install the [requirements](https://github.com/ddrrnn123/Omni-Seg/blob/main/Omni_seg_pipeline_gpu/requirements.txt)

4. Install [apex](https://github.com/NVIDIA/apex):
    ~~~
    cd Omni_Seg/Omni_seg_pipeline_gpu/apex
    python3 setup.py install
    cd ..
    ~~~

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