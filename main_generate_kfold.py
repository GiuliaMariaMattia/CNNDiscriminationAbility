#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Last updated version November 2023

@author: Giulia Maria Mattia

GNU Affero General Public License v3.0

Script to generate the 10-fold cross validation scheme and hold-out set
"""

# ***********************************************
# *------------------SCRIPT--------------------*
# ***********************************************

# ---------> Allocate memory
seed_value=1331
import os
os.environ['PYTHONHASHSEED']=str(seed_value)
import random as python_random
python_random.seed(seed_value)
import numpy as np
np.random.seed(seed_value)
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
sess=tf.keras.backend.get_session()
tf.keras.backend.clear_session()
sess.close()
tf.random.set_random_seed(seed_value)
tf.set_random_seed(seed_value)
config = tf.ConfigProto()
config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
config.gpu_options.per_process_gpu_memory_fraction = 0.7
set_session(tf.Session(graph=tf.get_default_graph(),config=config))


# ***********************************************
# *--------------IMPORT MODULES-----------------*
# ***********************************************
import nibabel as nib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn
import h5py
from keras.models import Sequential, Model
from keras.layers import Concatenate, Input, concatenate
import time
import csv
from scipy import stats
from keras.callbacks import ModelCheckpoint
from apmap_modules import*
from nilearn import plotting
import matplotlib.pyplot as plt
from keras.models import Model
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit
import scipy.ndimage as ndimage
from time import gmtime, strftime,localtime
import keras


# ***********************************************
# *--------------------MAIN---------------------*
# ***********************************************

if __name__ == "__main__":

    nb_param = len(sys.argv)
    if nb_param == 1:
        brain_region = input('>>> Enter altered brain regions (e.g. cerebellum, putamen, cerebellum_putamen) : ')
        id_folder = input('>>> Enter folder containing the APMaps : ') #e.g. perc0.75-ptile75
        size_split = input(" >>> Enter size of test and validation sets (e.g. 0.2,0.1) : ")

    else:
        brain_region = sys.argv[1]
        id_folder=sys.argv[2]
        size_split = sys.argv[3]

    #Check model name
    # if '3' not in model_name:
    #     raise RuntimeError("Wrong input model name")

    # ---------> load data
    id_image_set = 'MD'
    image_set = "Set_1"
    path_data = os.path.join(os.getcwd().replace("scripts","images"), image_set,id_image_set, "raw_data")
    data_type=path_data.split('/')[-1]
    label = 0

    patients_original =store_patients_data(data_path, mri_type, label)

    print('\n' + '>>> Original patients loaded from: ', path_data)
    list_synth=[]
    folder_synth_images=os.path.join(path_data.replace("raw_data","apmaps"),brain_region,id_folder)

    print("\n\n --> Images loaded from : " , folder_synth_images)
    list_synth.append(folder_synth_images)
    patients_synth = []
    patients_data = []
    Y_synth=[]
    label=1
    for folder in list_synth:
        # for idf in id_folder:
         #    if idf in folder:
             if id_folder in folder:
                 print("*********************************************")
                 print(' > Assigning label to ', folder)
                 full_path= os.path.join(folder_synth_images, folder)
                 patients = store_patients_data_ca(full_path, id_image_set, label)
                 # show_slices_patient(patients[0], folder_results=None)
                # patients_synth.append(patients)
                 patients_synth = patients_synth + patients
                # label += 1
    label += 1
    patients_data =  patients_original+ patients_synth

    list_pat=[patients_data[i].numPatient for i in range(len(patients_data))]
    list_age=[patients_data[i].age for i in range(len(patients_data))]
    len(np.unique(list_pat))
    num_classes = label #label #counting from 0

    Y_cat,Y= create_categorical_labels(patients_data, num_classes)

    if not len(np.unique(Y)) == label:
        print('ERROR: incorrect number of labels')

    # /*______________Check data______________*/
    patients_img=np.array([patient.img for patient in patients_data])
    patients_img[patients_img<0] = 0
    #for i in range(len(patients_img)):
    if np.min(patients_img) != 0:
       raise RuntimeError("Data scaled incorrectly in image {} --> range [{},{}]".format(i, np.min(patients_img) , np.max(patients_img)))
        # else:
    print(" >>> Intensity range [{:.2f},{:.5f}]".format(np.min(patients_img) , np.max(patients_img) ))

    rs =  40  #random state
    print(">>> Training, validation and test sets creation... --> random_state = ", rs)

    # /*______________Training and test set creation______________*/
    size_test=float(size_split.split(",")[0])
    size_val=float(size_split.split(",")[-1])
    print(">>> Training and test set creation... --> random_state = ", rs)
    x_train_val, x_test, y_train_val, y_test = train_test_split(patients_img, Y_cat, test_size=size_test, random_state=rs, stratify=Y) # patients_scaled = normalize_dataset(patients_data) if
    ax_train_val, ax_test, ay_train_val, ay_test = train_test_split(patients_data, Y_cat, test_size=size_test, random_state=rs, stratify=Y)
    print("Test patients : ", [patient.numPatient for patient in ax_test])

    x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=size_val, random_state=rs, stratify=y_train_val)
    bx_train,bx_val,by_train, by_val = train_test_split(ax_train_val,ay_train_val, test_size=size_val, random_state=rs, stratify=y_train_val)
    print("Validation patients : ", [patient.numPatient for patient in bx_val])
    numPat0 = [patient.numPatient for patient in ax_train_val]
    print("\n >>> TRAINING:  Number of samples per class: ", np.sum(y_train,axis=0))
    print(" >>> VALIDATION: Number of samples per class: ", np.sum(y_val,axis=0))
    print(" >>> TEST: Number of samples per class: ", np.sum(y_test,axis=0))

    #order x_train_val according to numpatient
    numPat_train_val = [patient.numPatient for patient in ax_train_val]
    ax_train_val=[ax_train_val[i] for i in sorted(range(len(numPat_train_val)), key=numPat_train_val.__getitem__)]
    ay_train_val=[ay_train_val[i] for i in sorted(range(len(numPat_train_val)), key=numPat_train_val.__getitem__)]
    x_train_val=[x_train_val[i] for i in sorted(range(len(numPat_train_val)), key=numPat_train_val.__getitem__)]
    y_train_val=[y_train_val[i] for i in sorted(range(len(numPat_train_val)), key=numPat_train_val.__getitem__)]
    numPat = [patient.numPatient for patient in ax_train_val]

    val_index=[]
    n_splits=10
    rs_fold=9

    sss = StratifiedKFold(n_splits,  random_state=rs_fold,shuffle=True)
    class_pat=[patient.label for patient in ax_train_val]
    kfold=sss.get_n_splits(x_train_val, class_pat)
    print("Number of folds  :  ",kfold)
    perf_folder=os.path.join(os.getcwd().replace("scripts","results"),"performance")
    os.makedirs(perf_folder, exist_ok=True)
    num_pat=sorted([patient.numPatient for patient in ax_train_val])

    fig, ax = plt.subplots()
    plot_cv_indices(sss, num_pat,class_pat, ax, n_splits)
    plt.savefig(os.path.join(perf_folder,"shuffle_{}fold_rs{}.png".format(n_splits,rs_fold)),bbox_inches='tight')
    plt.show()
    kfold_folder=os.path.join(os.getcwd().replace("scripts","results"),"kfold","indeces_rs{}".format(rs_fold))
    os.makedirs(kfold_folder, exist_ok=True)

    firstIteration=True
    k=0
    list_fold=[]
    for train_index, test_index in sss.split(x_train_val, class_pat):
        # print("TRAIN:", train_index, "\nTEST:", test_index)
        list_val=[ax_train_val[n].numPatient for n in test_index]
        print("Fold ", k)
        print("Samples in validation set = ", len(test_index))
        # if np.sum(np.array(list_val)>100)!=len(test_index)/2:
        #     raise RuntimeError("Imbalanced classes")
            # print()
        if firstIteration:
            val_index = test_index
            firstIteration=False
        else:
            val_index = np.concatenate(([val_index , test_index ]))
        X_train, X_val = np.array([x_train_val[i] for i in train_index]), np.array([x_train_val[i] for i in test_index])
        aX_train, aX_val = np.array([ax_train_val[i] for i in train_index]), np.array([ax_train_val[i] for i in test_index])
        y_train, y_val = np.squeeze(np.array([ay_train_val[i] for i in train_index])), np.squeeze(np.array([ay_train_val[i] for i in test_index]))
        print(">>> TRAINING:  Number of samples per class: ", np.sum(y_train,axis=0))
        print(" >>> VALIDATION: Number of samples per class: ", np.sum(y_val,axis=0))
        list_fold.append([k,np.array(train_index),np.array(test_index),np.max(X_train)])
        print("MAX in training", np.max(X_train))

        np.savetxt(os.path.join(kfold_folder,"val_index_rs{}_fold{}.txt".format(rs_fold,k)),test_index, delimiter=',',fmt='%i')
        np.savetxt(os.path.join(kfold_folder,"train_index_rs{}_fold{}.txt".format(rs_fold,k)),train_index, delimiter=',',fmt='%i') 
        np.savetxt(os.path.join(kfold_folder,"numpatient_val_index_rs{}_fold{}.txt".format(rs_fold,k)),[num_pat[i] for i in test_index], delimiter=',',fmt='%i')
        np.savetxt(os.path.join(kfold_folder,"numpatient_train_index_rs{}_fold{}.txt".format(rs_fold,k)),[num_pat[i] for i in train_index], delimiter=',',fmt='%i') 
        k += 1

    if len(np.unique(val_index)) != len(y_train_val):
         raise RuntimeError("Repetition of samples in  k fold")

    df_fold=pd.DataFrame(list_fold,columns =['Fold',"Train_index","Val_index","Train_maxValue"])

    df_fold.to_csv(os.path.join(kfold_folder,"cv_{}fold_rs{}.csv".format(kfold,rs_fold)),sep='\t')

    keras.backend.clear_session()
