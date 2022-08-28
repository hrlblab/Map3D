# Develop

## Installation

We used PyTorch 1.9.1 on [Ubuntu 22.04 LTS](https://releases.ubuntu.com/22.04/) with [Anaconda](https://www.anaconda.com/download) Python 3.7.

1. [Optional but recommended] Create a new Conda environment. 

    ~~~
    conda create --name Map3D python=3.7
    ~~~
    
    And activate the environment.
    
    ~~~
    conda activate Map3D
    ~~~

2. Clone the Map3D repo

3. Install the [requirements](https://github.com/hrlblab/Map3D/blob/main/Map3D-pipeline/requirements.txt)

4. Install [ANTs](https://github.com/ANTsX/ANTs):
    ~~~
    cd Map3D/Map3D-pipeline/ANTS
    sh installANTs.sh
    cd ..
    ~~~

## Run Map3D Registration pipeline locally without docker
- The entire pipeline is at the [Map3D-pipeline](Map3D-pipeline/) folder
- Create an empty folder in the [Map3D-pipeline](Map3D-pipeline/) folder and name it as "input_png". Put folders that contain PNG files into "input_png" folder. For guidance and instruction for input data format requirement and data arrangement, please refer to [DATA.md](https://github.com/hrlblab/Map3D/blob/main/DATA.md).
- Run python scripts in [Map3D-pipeline](Map3D-pipeline/) folder as following orders: 
  1. [Step1_superglue.py](https://github.com/hrlblab/Map3D/blob/main/Map3D-pipeline/Step1_superglue.py)
  2. [Step2_ApplySGToMiddle.py](https://github.com/hrlblab/Map3D/blob/main/Map3D-pipeline/Step2_ApplySGToMiddle.py)
  3. [Step3_matrix_npytomat.py](https://github.com/hrlblab/Map3D/blob/main/Map3D-pipeline/Step3_matrix_npytomat.py)
  4. [Step4_SuperGlue+ANTs.py](https://github.com/hrlblab/Map3D/blob/main/Map3D-pipeline/Step4_SuperGlue%2BANTs.py)
  5. [Step5_BigRecon_moveAllslicesToMiddle.py](https://github.com/hrlblab/Map3D/blob/main/Map3D-pipeline/Step5_BigRecon_moveAllslicesToMiddle.py)
- Note that [Step2_ApplySGToMiddle.py](https://github.com/hrlblab/Map3D/blob/main/Map3D-pipeline/Step2_ApplySGToMiddle.py), [Step4_SuperGlue+ANTs.py](https://github.com/hrlblab/Map3D/blob/main/Map3D-pipeline/Step4_SuperGlue%2BANTs.py), and [Step5_BigRecon_moveAllslicesToMiddle.py](https://github.com/hrlblab/Map3D/blob/main/Map3D-pipeline/Step5_BigRecon_moveAllslicesToMiddle.py) require an argument. Example can be found in "Map3D Registration Demo" section at README.md.
- The output will be stored at "output" folder under [Map3D-pipeline](Map3D-pipeline/) directory.