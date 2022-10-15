# Map3D-Registration: An End-to-end Pipeline for 3D Slide-wise Multi-stain Renal Pathology Registration

### [[Accelerated Pipeline Docker]](https://hub.docker.com/r/peize/map3d)[[Project Page]](https://github.com/hrlblab/Map3D)[[IEEE TMI Paper]](https://arxiv.org/pdf/2006.06038.pdf)[[SPIE 2022 Paper]](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12039/120390F/Dense-multi-object-3D-glomerular-reconstruction-and-quantification-on-2D/10.1117/12.2611957.full?SSO=1)[[SPIE 2023 Paper]](https://github.com/hrlblab/Map3D)<br />


This is the official implementation of Map3D-Registration: An End-to-end Pipeline for 3D Slide-wise Multi-stain Renal Pathology Registration

![Overview1](https://github.com/hrlblab/Map3D/blob/main/Figure/IEEE%20TMI%20Map3D.png)<br />
![Overview2](https://github.com/hrlblab/Map3D/blob/main/Figure/pipeline.png)<br />

**IEEE Transactions on Medical Imaging Paper** <br />
> [Map3D: Registration Based Multi-Object Tracking
on 3D Serial Whole Slide Images](https://arxiv.org/pdf/2006.06038.pdf) <br />
> Ruining Deng, Haichun Yang, Aadarsh Jha, Yuzhe Lu, Peng Chu, Agnes B. Fogo, and  Yuankai Huo. <br />

**SPIE 2022 Paper** <br />
> [Dense multi-object 3D glomerular reconstruction and quantification on 2D serial section whole slide images](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12039/120390F/Dense-multi-object-3D-glomerular-reconstruction-and-quantification-on-2D/10.1117/12.2611957.full?SSO=1)<br />
>Ruining Deng, Haichun Yang, Zuhayr Asad, Zheyu Zhu, Shiru Wang, Lee E. Wheless, Agnes B. Fogo, Yuankai Huo.<br />


**SPIE 2023 Paper** <br />
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
sudo docker pull peize/map3d
```
#### Run Map3D Pipeline
You can run the following commands to run Map3D Registration pipeline. You may change the `input_dir` and the list of indexes, and then you will have the final segmentation results in `output_dir`. Please refer to [DATA.md](https://github.com/hrlblab/Map3D/blob/main/DATA.md) for input data format requirement and data arrangement.
```
# you need to specify the input directory. 
export input_dir=/home/input_dir   

# set output directory
export output_dir=$input_dir/output

# run the docker
sudo nvidia-docker run -it --rm -v $input_dir:/INPUTS -v $output_dir:/OUTPUTS peize/map3d

# Enter a comma seperated list of indexes to indicate which image should be used as the middle section image in each case
2,3,5
```
## Run Pipeline Locally without Docker
Please refer to [Develop.md](https://github.com/hrlblab/Map3D/blob/main/DEVELOP.md) for instructions of running Map3D Registration pipeline locally.

## Data
An example dataset of needle biopsy tissue samples for the pipeline can be found [here](https://vanderbilt.box.com/s/3lrrwxrk9wpw7gkgupfw1pzcgvuyv8uz). This dataset contains two different cases and each includes five .PNG files with 10X magnification.

Another example dataset of large tissue samples can also be found [here](https://vanderbilt.box.com/s/9vjn2h50al2xn7f0f0spknoumlu0hvs2). This dataset contains a single case of seven .PNG files with 10X magnification.


## Map3D Registration Demo
### Needle Biopsy Tissue Samples
<img src="https://github.com/hrlblab/Map3D/blob/main/Figure/Needle_figure.png" height="700">

Below is an example input of serial section WSIs of needle biopsy tissue samples. These images are contained in our demo dataset, which can be found in the "Data" Section above.

<img src='Figure/before.png' align="center" height="230px"> 

- The entire pipeline is at the [Map3D-pipeline](Map3D-pipeline/) folder
- Create an empty folder in the [Map3D-pipeline](Map3D-pipeline/) folder and name it as "input_png". Put folders that contain 10X magnification PNG files into "input_png" folder. For guidance and instruction for input data format requirement and data arrangement, please refer to [DATA.md](https://github.com/hrlblab/Map3D/blob/main/DATA.md).
- Use our docker image as shown in "Quick Start" Section or run the python scripts as following orders:
  1. [Step1_superglue.py](https://github.com/hrlblab/Map3D/blob/main/Map3D-pipeline/Step1_superglue.py)
  ```
  python Step1_superglue.py
  ```
  2. [Step2_ApplySGToMiddle.py](https://github.com/hrlblab/Map3D/blob/main/Map3D-pipeline/Step2_ApplySGToMiddle.py)
  ```
  python Step2_ApplySGToMiddle.py --middle_images 2,3,2
  ```
  3. [Step3_matrix_npytomat.py](https://github.com/hrlblab/Map3D/blob/main/Map3D-pipeline/Step3_matrix_npytomat.py)
  ```
  python Step3_matrix_npytomat.py
  ```
  4. [Step4_SuperGlue+ANTs.py](https://github.com/hrlblab/Map3D/blob/main/Map3D-pipeline/Step4_SuperGlue%2BANTs.py)
  ```
  python Step4_SuperGlue+ANTs.py --middle_images 2,3,2
  ```
  5. [Step5_BigRecon_moveAllslicesToMiddle.py](https://github.com/hrlblab/Map3D/blob/main/Map3D-pipeline/Step5_BigRecon_moveAllslicesToMiddle.py)
  ```
  python Step5_BigRecon_moveAllslicesToMiddle.py --middle_images 2,3,2
  ```
- The output will be stored at "output" folder under [Map3D-pipeline](Map3D-pipeline/) directory.

If set up correctly, the output for "no1" should look like

<img src='Figure/after.png' align="center" height="230px"> 

### Large Tissue Sample
Below is an example input of serial section WSIs of large tissue samples. These images are also contained in our demo dataset, which can be found in the "Data" Section above.
<img src='Figure/before_scn.png' align="center" height="230px">

Follow the same steps as for needle biopsy tissue samples. The output should look like 
<img src='Figure/after_scn.png' align="center" height="230px"> 


## Citation
```
@article{deng2021map3d,
title={Map3D: registration based multi-object tracking on 3D serial whole slide images},
author={Ruining Deng, Haichun Yang, Aadarsh Jha, Yuzhe Lu, Chu Peng, Agnes B. Fogo, and Yuankai Huo},
journal={IEEE Transactions on Medical Imaging},
year={2021},
publisher={IEEE},
url={https://arxiv.org/pdf/2006.06038.pdf}
}
```
