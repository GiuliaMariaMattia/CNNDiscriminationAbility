# -*- coding: utf-8 -*-
"""
Last updated version November 2023

@author: Giulia Maria Mattia

"""

# ***********************************************
# *-----------------MODULES---------------------*
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
from skimage import morphology
import scipy.ndimage as ndimage


class Patient:
    def __init__(self, numPatient, label, idPatient):
        self.numPatient = numPatient
        self.label = label
        self.idPatient = idPatient 

    #Add images to the class
    def add_img(self, img):
        self.img = img
    
    #Add filename
    def add_filename(self, filename):
        self.filename = filename

    #Add image info to the class
    def add_header(self, header):
        self.header = header

    def add_affine(self, affine):
            self.affine = affine

def store_patients_data(data_path, mri_type,label):
    """ Function to store patients' data 
    @param (str) data_path : path containing images to be loaded
    @param (str) mri_type : keyword in each image filename corresponding to the MRI type (e.g. MD)
    @param (int) label: class to assign
    @return (list) patients_data: list of loaded data in the Patient class object
    """
    patients_data = []

    """ Split file names using mri_type
     [ID_patient]_[mri_type]_[image_specs].nii.gz
    """
    numPatient = 0 #patients' counter
    for file in sorted(os.listdir(data_path)): #patients loaded in alphabetical order
        full_file = os.path.join(data_path,file)   
        id_Patient = file.split(mri_type)[0]
        patients_data.append(Patient(numPatient, label,  id_Patient))
        patients_data[-1].add_filename(file)
        patients_data[-1].add_img(nib.load(full_file).get_fdata())
        patients_data[-1].add_header(nib.load(full_file).header)
        patients_data[-1].add_affine(nib.load(full_file).affine)
        numPatient += 1
        # print("Patient {} | ID : {} | Label {}".format(numPatient, idPatient, label))
        if np.sum(patients_data[-1].img) == 0:
            print('Null image for ', patients_data[-1].filename)
                
    return patients_data


def create_apmaps(patients_data, apmaps_path,apmaps_info,mask_path):
      """

      Parameters
      ----------
      patients_data : list 
          list of objects of class Patient containing the image to alter (patients_data[i].img)
      apmaps_path : str
          path to save the altered images 
      apmaps_info : list
          apmaps_info[0]: percentile of the histogram used as threshold to modify images
          apmaps_info[1]: percentage to increase value with respect to the original value
      Returns
      -------
      None.

      """
      brain_region_data=[]
      brain_region_perc=[]
      modified_patients=[]
      
      #Load region mask
      template =load_img(mask_path)
      template=np.squeeze(load_img(mask_path).get_fdata())
      pixel_mask=(np.where(template.ravel()==1))
      pixel_mask[0].shape
      print(" \n--->>> Non-zero voxels in mask : ", np.where(template.ravel()==1)[0].shape)

      percentile_hist=apmaps_info[0]
      percentage=apmaps_info[1]

      print('\n Patient # ')
      for i in range(len(patients_data)):
          # print("************************************************************")
          # print(' Patient {}'.format( patients[i].numPatient))
          print(patients_data[i].numPatient, end=',', flush=True)
    
          #Consider i-th patient
          tmp = np.array(patients_data[i].img)
          filename=patients_data[i].filename

          #Multiply created mask and original image
          tmp_reg= np.multiply(tmp,np.squeeze(template))

          #Compute percentile of the image (considering non zero values)
          brain_region_data.append(tmp_reg[np.where(tmp_reg!=0)])
          percentile_thresh=np.percentile(brain_region_data[-1],percentile_hist)
          # print(">>> Percentile {}th : {}".format(percentile_hist, percentile_thresh))

          #Modify region according to n-th percentile of brain region mask (zero values excluded)
          brain_region_perc.append(np.where(np.logical_and(tmp_reg>0, tmp_reg<=percentile_thresh),tmp_reg*percentage,0))

          #Compute the final altered image (APMap)
          tmp_mod = brain_region_perc[-1] + tmp

          #Set maximum value as the one of the original image
          modified_patients.append(np.where(tmp_mod>np.max(tmp),np.max(tmp),tmp_mod))

          #Save altered image into a nifti file
          output_image = nib.Nifti1Image(modified_patients[-1], patients_data[i].affine, patients_data[i].header)
          output_image.to_filename(os.path.join(apmaps_path,patients_data[i].filename))
              
      return modified_patients
  

def create_apmaps_size_change(patients_data, apmaps_path,apmaps_info,mask_path,type_morph,brain_reg):
      """

      Parameters
      ----------
      patients_data : list 
          list of objects of class Patient containing the image to alter (patients_data[i].img)
      apmaps_path : str
          path to save the altered images 
      apmaps_info : list
          apmaps_info[0]: percentile of the histogram used as threshold to modify images
          apmaps_info[1]: percentage to increase value with respect to the original value
      type_morph : str
          type of morphological operation to apply ("e" for erosion and "d" for dilation)
      Returns
      -------
      None.

      """
      brain_region_data=[]
      brain_region_perc=[]
      modified_patients=[]
      
      #Load region mask and perform morphological operation (morph_size and brain_reg can be changed to test other regions or morphological operations)
      template =load_img(mask_path)
      template=np.squeeze(load_img(mask_path).get_fdata())
      if type_morph=="d" and brain_reg=="putamen":
          morph_size=7
          template=ndimage.morphology.binary_dilation(template,iterations=morph_size)
      elif type_morph=="e" and brain_reg=="cerebellum":
          morph_size=6
          template=ndimage.morphology.binary_erosion(template,iterations=morph_size)
          template=morphology.remove_small_objects(template,3)
      nifti_image = nib.Nifti1Image(template, load_img(mask_path).affine, load_img(mask_path).header)
      nifti_image.to_filename("{}-{}-iter{}.nii.gz".format(type_morph,brain_reg,morph_size))
      pixel_mask=(np.where(template.ravel()==1))
      pixel_mask[0].shape
      print(" \n--->>> Non-zero voxels in mask : ", np.where(template.ravel()==1)[0].shape)

      percentile_hist=apmaps_info[0]
      percentage=apmaps_info[1]

      print('\n Patient # ')
      for i in range(len(patients_data)):
          # print("************************************************************")
          # print(' Patient {}'.format( patients[i].numPatient))
          print(patients_data[i].numPatient, end=',', flush=True)
    
          #Consider i-th patient
          tmp = np.array(patients_data[i].img)
          filename=patients_data[i].filename

          #Multiply created mask and original image
          tmp_reg= np.multiply(tmp,np.squeeze(template))

          #Compute percentile of the image (considering non zero values)
          brain_region_data.append(tmp_reg[np.where(tmp_reg!=0)])
          percentile_thresh=np.percentile(brain_region_data[-1],percentile_hist)
          # print(">>> Percentile {}th : {}".format(percentile_hist, percentile_thresh))

          #Modify region according to n-th percentile of brain region mask (zero values excluded)
          brain_region_perc.append(np.where(np.logical_and(tmp_reg>0, tmp_reg<=percentile_thresh),tmp_reg*percentage,0))

          #Compute the final altered image (APMap)
          tmp_mod = brain_region_perc[-1] + tmp

          #Set maximum value as the one of the original image
          modified_patients.append(np.where(tmp_mod>np.max(tmp),np.max(tmp),tmp_mod))

          #Save altered image into a nifti file
          output_image = nib.Nifti1Image(modified_patients[-1], patients_data[i].affine, patients_data[i].header)
          output_image.to_filename(os.path.join(apmaps_path,patients_data[i].filename))
              
      return modified_patients