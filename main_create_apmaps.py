# -*- coding: utf-8 -*-
"""
Last updated version November 2023

@author: Giulia Maria Mattia

Script to create the altered parametric maps (APMaps) in the monoregion version.
The biregion APMaps can be obtained by modyfing one region first and then use these monoregion APMaps as base to modify the second region,
instead of using the original parametric maps

"""

# ***********************************************
# *--------------IMPORT MODULES-----------------*
# ***********************************************

import os
from os.path import isdir
from glob import glob
from pathlib import Path
import pandas as pd
from nilearn.image import load_img, resample_to_img
from nilearn import plotting
import numpy as np
from nibabel.nifti1 import Nifti1Image
import matplotlib.pyplot as plt
import nibabel as nib
from apmap_modules import*


# ***********************************************
# *--------------------MAIN---------------------*
# ***********************************************
      

# if __name__ == "__main__":
#    nb_param = len(sys.argv)
#    if nb_param == 1:
#        brain_region = input(">>> Enter brain region(s) separated by comma : ")
#        mri_type = input('>>> Enter MRI type: ')
#        apmaps_info = input('>>> Enter percentile and percentage separate by comma (e.g. 75,0.99): ')
#        image_set = input('>>> Enter name of image set : ')
#    else:
#        brain_region = sys.argv[1]
#        mri_type = sys.argv[2]
#        apmaps_info = sys.argv[3]
#        image_set = sys.argv[4]

#Name of the region to modify
brain_region="cerebellum"

#To apply morphological operation on region mask ("e": erosion, "d":dilation, "": none)
morph_op_type = ""

#Type of MRI data
mri_type="MD"

#apmaps_info: Parameters to create the APMaps [n-th percentile, percentage]. 
#The n-th percentile is needed as a threshold determined by considering the intensity distribution to avoid image saturation effect.
#The percentage indicates the intensity increase to apply.
apmaps_info = [75,0.75]

#Name of the considered dataset
image_set = "Set_1"

#Path containing images to modify
data_path = os.path.join(os.getcwd().replace("scripts","images"), image_set,mri_type, "raw_data")
 
print("Data loaded from \n", data_path)
data_type=data_path.split('/')[-1]
label = 0
list_patients = store_patients_data(data_path, mri_type, label)


#Path containing the modified images (i.e. the APMaps)
if morph_op_type == "":
    apmaps_path=os.path.join(data_path.replace("raw_data","apmaps"),brain_region,"perc{:.2f}-ptile{}".format(apmaps_info[1],apmaps_info[0]))
else:
    apmaps_path=os.path.join(data_path.replace("raw_data","apmaps"),"-".join([morph_op_type,brain_region]),"perc{:.2f}-ptile{}".format(apmaps_info[1],apmaps_info[0]))

os.makedirs(apmaps_path, exist_ok=True)
print('Folder to save APMaps : \n',apmaps_path)

#Enter path containing the mask of the region to modify
mask_file = "{}_mask.nii.gz".format(brain_region)
mask_path= os.path.join(os.getcwd().replace("scripts","images"), "masks",mask_file)

if morph_op_type == "":
#Create the APMaps by changing the intensity of the region
    apmaps_list = create_apmaps( list_patients, apmaps_path,apmaps_info,mask_path)
else:
#Create the APMaps by changing intensity and size of the region 
    apmaps_list =create_apmaps_size_change( list_patients, apmaps_path,apmaps_info,mask_path,morph_op_type,brain_region)

print('>>> Creation of APMaps completed')
