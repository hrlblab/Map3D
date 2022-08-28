# Data

This document provides tutorials to develop Map3D Registration.

## New data
Basically there are three steps:

- Annotate the WSI rectangularly. Recommend to use ImageScope and save the .xml file for annotation information. 
- Convert the svs file into PNG files and saved into 10X magnification. Please refer to [scnToPng.py](https://github.com/hrlblab/Map3D/blob/main/Map3D-pipeline/scnToPng.py) for an example of converting .svs files into 10X magnification .png files.
- Put PNG files from the same series of WSI into the same folder and name these folders as "no1", "no2", and etc.
- Create one empty folder to be the input folder. Put "no1", "no2", ... into this input folder.

