#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Last updated version November 2023

@author: Giulia Maria Mattia

Script to implement a 3D CNN trained with OPMaps and APMaps to determine CNN behavior according to the
specific input features (e.g. intensity and size of the altered region in the APMaps)

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
from sklearn.model_selection import train_test_split
import scipy.ndimage as ndimage
from time import gmtime, strftime,localtime
import keras

s = 3 #kernel size
lambda_regularizer = 5e-4
initializer = tf.keras.initializers.lecun_uniform(seed=seed_value)
# print(initializer)
# input("Proceed?")
def get_input_downsampling(input_img, filter_size, pool, n, depth):
    if keras.backend.image_data_format() == 'channels_first':
        axe = 1
    else:
        axe = -1

    for i in range(len(filter_size)):
        if i == 0:
            downsampling = keras.layers.Conv3D(filter_size[i],
                                            name='Conv3D_{}_{}_{}_{}'.format(depth, i, filter_size[i],n),
                                          kernel_size=(s,s,s),
                                          strides=(1, 1, 1),
                                          padding='valid',
                                          data_format=keras.backend.image_data_format(),
                                          kernel_regularizer=keras.regularizers.l2(lambda_regularizer),
                                          use_bias=True,
                                          kernel_initializer=initializer, #previously initializer
                                          bias_initializer='zeros')(input_img)
        else:
            downsampling = keras.layers.Conv3D(filter_size[i],
                                            name='Conv3D_{}_{}_{}_{}'.format(depth, i, filter_size[i],n),
                                          kernel_size=(s,s,s),
                                          strides=(1, 1, 1),
                                          padding='valid',
                                          data_format=keras.backend.image_data_format(),
                                          kernel_regularizer=keras.regularizers.l2(lambda_regularizer),
                                          use_bias=True,
                                          kernel_initializer=initializer,
                                          bias_initializer='zeros')(downsampling)
        downsampling = keras.layers.BatchNormalization(axis=axe,
                                                name='BN_{}_{}_{}'.format(i, depth, n),
                                                momentum=0.99,
                                                epsilon=1e-8,
                                                center=True,
                                                scale=True,
                                                beta_initializer='zeros',
                                                gamma_initializer='ones')(downsampling)
        downsampling = keras.layers.ELU(alpha=1.0,
                                                name='relu_{}_{}_{}'.format(i, depth, n))(downsampling)

    if pool:
        downsampling = keras.layers.AveragePooling3D(pool_size=(2,2,2), padding='valid', data_format=keras.backend.image_data_format(),
                                                name='pooling_{}_{}_{}'.format(i, depth, n))(downsampling)
       

    return downsampling


def get_out_branch(input,num_classes):
    if keras.backend.image_data_format() == 'channels_first':
        axe = 1
    else:
        axe = -1

    neurons=512
    out = keras.layers.Dense(neurons, use_bias=True,
                                kernel_initializer=initializer,
                                bias_initializer='zeros')(input)


    out = keras.layers.BatchNormalization(axis=axe,
                                            momentum=0.99,
                                            epsilon=1e-8,
                                            center=True,
                                            scale=True,
                                            beta_initializer='zeros',
                                            gamma_initializer='ones')(out)
    out = keras.layers.ELU(alpha=1.0)(out)

    out = keras.layers.Dropout(1-0.5,seed=seed_value)(out)

    out = keras.layers.Dense(neurons, use_bias=True,
                                kernel_initializer=initializer,
                                bias_initializer='zeros')(out)

    #
    out = keras.layers.BatchNormalization(axis=axe,
                                            momentum=0.99,
                                            epsilon=1e-8,
                                            center=True,
                                            scale=True,
                                            beta_initializer='zeros',
                                            gamma_initializer='ones')(out)
    out = keras.layers.ELU(alpha=1.0)(out)

    out = keras.layers.Dropout(1-0.75,seed=seed_value)(out)

    out = keras.layers.Dense(neurons, use_bias=True,
                                kernel_initializer=initializer,
                                bias_initializer='zeros')(out)


    out = keras.layers.BatchNormalization(axis=axe,
                                            momentum=0.99,
                                            epsilon=1e-8,
                                            center=True,
                                            scale=True,
                                            beta_initializer='zeros',
                                            gamma_initializer='ones')(out)
    out = keras.layers.ELU(alpha=1.0)(out)

    out = keras.layers.Dropout(1-0.75,seed=seed_value)(out)

    out = keras.layers.Dense(neurons, use_bias=True,
                                kernel_initializer=initializer,
                                bias_initializer='zeros')(out)


    out = keras.layers.BatchNormalization(axis=axe,
                                            momentum=0.99,
                                            epsilon=1e-8,
                                            center=True,
                                            scale=True,
                                            beta_initializer='zeros',
                                            gamma_initializer='ones')(out)
    out = keras.layers.ELU(alpha=1.0)(out)

    out = keras.layers.Dense(num_classes, use_bias=True,
                                kernel_initializer=initializer,
                                bias_initializer='zeros')(out)

    return out


def create_cnn_model(inputs_shapes, names,num_classes, path_save):
    if keras.backend.image_data_format() == 'channels_first':
        axe = 1
    else:
        axe = -1

    input = keras.layers.Input(shape=inputs_shapes)
    branch = get_input_downsampling(input, [32], True, names, 1)
    branch = get_input_downsampling(branch, [64], True, names, 2)
    branch = get_input_downsampling(branch, [128,128,128], True, names, 3)

    branch_flat = keras.layers.Flatten()(branch)

    out = get_out_branch(branch_flat,num_classes)

    out = keras.layers.Activation('softmax')(out)

    model = Model(input, out)

    model.summary()

    keras.utils.plot_model(model, to_file=os.path.join(path_save,names +'_model.png'), \
                     show_shapes=True, show_layer_names=True)
    return model


class DebugModel(keras.callbacks.Callback):
    def on_batch_end(self, batch, logs):
        print(keras.backend.eval(self.model.layers[-1].output))


def train_cnn_model(model, opt, batch_size, validation_split, shuffle, verbose, epochs,
                XTrain, YTrain, XTest, YTest, model_name, path_save,num_classes,nfold):
    start_time = time.time()
    print(">>>>> Training architecture ...")

    checkpointer = ModelCheckpoint(filepath=os.path.join(path_save,"best_model_fold{}.hdf5".format(nfold)),
                                   monitor = 'val_loss',
                                   verbose=1,
                                   save_best_only=True)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.25, patience=5, verbose=1, mode='min', cooldown=1, min_lr=1e-14)

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])


    history = model.fit(x=XTrain, y=YTrain,
              batch_size = batch_size,
              epochs = epochs,
              validation_data=(XTest, YTest),
              #validation_split=validation_split,
              shuffle=shuffle,
              verbose=verbose,
              callbacks=[checkpointer,reduce_lr])
    print('\n################################')
    print('Training time : ','--- {0:8.2f} min ---'.format((time.time() - start_time)/60))
    print('\n################################')
    print('Performance of last epoch on validation set with ' + model_name)
    predictions = model.predict(x=XTest)
    score = model.evaluate(x=XTest, y=YTest, verbose=verbose)
    print('Loss: ' + str(score[0]))
    print('Accuracy: ' + str(score[1]))

    # model.save(os.path.join(path_save, 'model_MD.h5'))
    del model


    return history, predictions, score



# ***********************************************
# *--------------------MAIN---------------------*
# ***********************************************

if __name__ == "__main__":

    nb_param = len(sys.argv)
    if nb_param == 1:
        brain_region = input('>>> Enter altered brain regions (e.g. cerebellum, putamen, cerebellum_putamen) : ')
        model_name = input('>>> Enter model name : ')
        id_folder =input('>>> Enter folder containing the APMaps : ') #e.g. perc0.75-ptile75
        flag_std=input(" >>> Normalization with maximum training set value (nt) ")
        flag_all=input(" >>> Entire data set in training ? y/n \t")
        size_split = input(" >>> Enter size of test and validation sets (e.g. 0.2,0.1) ? y/n \t")
        rep_num=input(" >>> Enter repetition number:  \t")
        nfold=input(" >>> Enter fold number for cross validation :  \t")
        rs_fold=input(" >>> Enter random state for fold generation :  \t")
    else:
        brain_region = sys.argv[1]
        model_name = sys.argv[2]
        id_folder=sys.argv[3]
        flag_std = sys.argv[4]
        flag_all = sys.argv[5]
        size_split = sys.argv[6]
        rep_num = sys.argv[7]
        nfold = sys.argv[8]
        rs_fold = sys.argv[9]

    #Check model name
    if '3' not in model_name:
        raise RuntimeError("Wrong input model name")

    # ---------> load data
    print("\n\n  ****************  Starting ITERATION ", rep_num)
    id_image_set = 'MD'
    image_set = "Set_1"
    path_data = os.path.join(os.getcwd().replace("scripts","images"), image_set,id_image_set, "raw_data")
    data_type=path_data.split('/')[-1]
    label = 0
    patients_original = store_patients_data(path_data, id_image_set, label)

    print('\n' + '>>> Original patients loaded from: ', path_data)
    list_synth=[]
    folder_synth_images=os.path.join(path_data.replace("raw_data","apmaps"),brain_region,id_folder)

    print("\n\n --> Images loaded from : " , folder_synth_images)
    list_synth.append(folder_synth_images)
    patients_synth=[]
    Y_synth=[]
    label=1
    for folder in list_synth:
        if id_folder in folder:
            print("*********************************************")
            print(' > Assigning label to ', folder)
            full_path= os.path.join(folder_synth_images, folder)
            patients = store_patients_data(full_path, id_image_set, label)
            patients_synth = patients_synth + patients
           
    label += 1
    patients_data =  patients_original+ patients_synth

    if flag_std == "nt":
        print("________NORMALIZATION with training set max value________")
        model_folder_id = "NORMtrain"
    else:
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Wrong input")

    list_pat=[patients_data[i].numPatient for i in range(len(patients_data))]
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

    kfold_folder=os.path.join(os.getcwd().replace("scripts","results"),"kfold","details_rs{}".format(rs_fold))
    df_fold=pd.read_csv(os.path.join(kfold_folder,"cv_10fold_rs{}.csv".format(rs_fold)),sep='\t',engine='python')
    df_fold=df_fold.loc[df_fold["Fold"]==int(nfold)]
    train_index=np.loadtxt(os.path.join(kfold_folder,"train_index_rs{}_fold{}.txt".format(rs_fold,nfold)),dtype=int)
    val_index=np.loadtxt(os.path.join(kfold_folder,"val_index_rs{}_fold{}.txt".format(rs_fold,nfold)),dtype=int)

    # /*______________Training and test set creation______________*/
    size_test=float(size_split.split(",")[0])
    size_val=float(size_split.split(",")[-1])
    print(">>> Training and test set creation... --> random_state = ", rs)
    x_train_val, x_test, y_train_val, y_test = train_test_split(patients_img, Y_cat, test_size=size_test, random_state=rs, stratify=Y) # patients_scaled = normalize_dataset(patients_data) if
    ax_train_val, ax_test, ay_train_val, ay_test = train_test_split(patients_data, Y_cat, test_size=size_test, random_state=rs, stratify=Y)
    print("Test patients : ", [patient.numPatient for patient in ax_test])

    #order according to patient number to match details for fold sets
    numPat_train_val = [patient.numPatient for patient in ax_train_val]
    ax_train_val=[ax_train_val[i] for i in sorted(range(len(numPat_train_val)), key=numPat_train_val.__getitem__)]
    ay_train_val=[ay_train_val[i] for i in sorted(range(len(numPat_train_val)), key=numPat_train_val.__getitem__)]
    x_train_val=[x_train_val[i] for i in sorted(range(len(numPat_train_val)), key=numPat_train_val.__getitem__)]
    y_train_val=[y_train_val[i] for i in sorted(range(len(numPat_train_val)), key=numPat_train_val.__getitem__)]
    numPat = [patient.numPatient for patient in ax_train_val]

    x_train, x_val = np.array([x_train_val[i] for i in train_index]), np.array([x_train_val[i] for i in val_index])
    bx_train, bx_val =[ax_train_val[i] for i in train_index],[ax_train_val[i] for i in val_index]
    y_train, y_val = np.squeeze(np.array([y_train_val[i] for i in train_index])), np.squeeze(np.array([y_train_val[i] for i in val_index]))

    print("\n >>> TRAINING:  Number of samples per class: ", np.sum(y_train,axis=0))
    print(" >>> VALIDATION: Number of samples per class: ", np.sum(y_val,axis=0))
    print(" >>> TEST: Number of samples per class: ", np.sum(y_test,axis=0))

    max_val = np.max(x_train)
    x_train = np.divide(x_train,max_val)
    x_val = np.divide(x_val,max_val)
    x_test = np.divide(x_test, max_val)

    print(" >>> TRAINING range: ", np.min(x_train), np.max(x_train))
    print(" >>> VALIDATION range: ", np.min(x_val), np.max(x_val))
    print(" >>> TEST range: ", np.min(x_test), np.max(x_test))



    # # /*______________Setting CNN______________*/
    epochs = 100
    pooling_type ='AVG'
    batch_size = 8 
    parameters = "epochs{}_bs{}_pt{}_classes{}_rs{}_rsfold{}_tes{:.2f}_val{:.2f}".format(epochs,batch_size,pooling_type,num_classes,rs,rs_fold,size_test,size_val)
    if  == "y":
        parameters += "_all"
    model_name=model_name.replace("_","-")
    folder_results=os.path.join(os.path.dirname(kfold_folder),model_name,parameters,model_folder_id,brain_region,id_folder)
    os.makedirs(folder_results, exist_ok=True)

    print("\n\n <<<<<<< Results will be saved in ")
    print(folder_results)


  # /*______________Training and test set creation______________*/
    image_size = np.array(patients_data[0].img).shape
    print("Image size : ", image_size)
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    x_val=np.array(x_val)
    if keras.backend.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, image_size[0], image_size[1], image_size[2])
        x_test = x_test.reshape(x_test.shape[0], 1, image_size[0], image_size[1], image_size[2])
        x_val = x_val.reshape(x_val.shape[0], 1, image_size[0], image_size[1], image_size[2])
        input_shape = (1, image_size[0], image_size[1], image_size[0])
    else:
        x_train = x_train.reshape(x_train.shape[0], image_size[0], image_size[1], image_size[2], 1)
        x_test = x_test.reshape(x_test.shape[0], image_size[0], image_size[1], image_size[2], 1)
        x_val = x_val.reshape(x_val.shape[0], image_size[0], image_size[1], image_size[2], 1)
        input_shape = (image_size[0], image_size[1], image_size[0], 1)



    # /*______________Training CNN______________*/
    Size_train_val_test=[len(x) for x in [y_train,y_val,y_test]]
    print("<<< Total number of classes  ", num_classes)
    model = create_cnn_model(input_shape, 'MD', num_classes,folder_results)
    lr=5e-5
    shuffle = True
    validation_split = 0
    verbose = 2
    print(" --> Initial learning rate : ", lr)
    opt = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.1, amsgrad=True)
    hist, predictions, score = train_cnn_model(model, opt,batch_size, validation_split, shuffle, verbose, epochs,
                x_train, y_train, x_val, y_val, model_name, folder_results,num_classes,nfold)
    perf_all=zip(*[hist.history['acc'],hist.history['val_acc'],hist.history['loss'],hist.history['val_loss']])

    with open(os.path.join(folder_results, "training_performance_fold{}.csv".format(nfold)),"w") as perf:
        fieldnames=["Acc","Val_acc","Loss","Val_loss"]
        wr=csv.writer(perf,delimiter="\t")
        wr.writerow(fieldnames)
        for item in perf_all:
             wr.writerow(item)
    np.savetxt(os.path.join(folder_results,"perf_{}_fold{}.txt".format(data_type,nfold)), hist.history['acc'], delimiter=',',fmt = "%.3f" )

    best_model_weights=os.path.join(folder_results,"best_model_fold{}.hdf5".format(nfold))
    if os.path.exists(best_model_weights):
        best_model=keras.models.load_model(best_model_weights)
        print("\n\n>>> Loading weights of best model... ")

    score = best_model.evaluate(x_test, y_test, verbose=0)
    print("Predictions  : ", np.sum(np.round(best_model.predict(x_test)),axis=0))
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    y_test_int=from_categorical_to_int_labels(y_test)
    y_pred_int=from_categorical_to_int_labels(np.round(best_model.predict(x_test)))

    cnf_matrix = confusion_matrix(y_test_int,y_pred_int)
    np.set_printoptions(precision=2)

    cnf_matrix_norm = confusion_matrix(y_test_int,y_pred_int, normalize='true')
    print("Labels ", y_test_int)
    y_diff=y_pred_int-y_test_int
    mismatched_fp=[ax_test[i].numPatient for i in np.where(y_diff>0)[0]]
    mismatched_fn=[ax_test[i].numPatient for i in np.where(y_diff<0)[0]]
    cnf_matrix = confusion_matrix(y_test_int,y_pred_int)
    print("CM : {}".format(np.round(cnf_matrix_norm, decimals=3).ravel()))
    print(">>> FP : Total = ",len(mismatched_fp)    )
    print(mismatched_fp)
    print(">>> FN : Total = " ,len(mismatched_fn)   )
    print(mismatched_fn)


    file_training = os.path.join(os.path.dirname(kfold_folder),model_name,"training_details_folds_{}.csv".format(brain_region))
    row_contents=[ strftime("%Y-%m-%d %H:%M:%S", localtime()),model_name,parameters,brain_region,model_folder_id,id_folder,Size_train_val_test,np.mean(hist.history['acc']),np.std(hist.history['acc']),np.mean(hist.history['loss']),np.std(hist.history['loss']),np.mean(hist.history['val_acc']),np.std(hist.history['val_acc']),np.mean(hist.history['val_loss']),np.std(hist.history['val_loss']),score[1],score[0],nfold,rs_fold,mismatched_fp,mismatched_fn]
    if os.path.exists(file_training):
        append_list_as_row(file_training, row_contents)
    else:
        with open(file_training,"w") as file_t:
            fieldnames=["Date","Model","Setting","Brain_region","Scaling","Training", "Size_train_val_test","Mean_train_acc","SD_train_acc","Mean_train_Loss","SD_train_loss","Mean_Val_acc","SD_Val_acc","Mean_Val_Loss","SD_Val_loss","Test_acc","Test_loss","Fold","Random state","FP","FN"]
            #perf.write(fieldnames)
            wr=csv.writer(file_t,delimiter="\t")
            wr.writerow(fieldnames)
            wr.writerow(row_contents)

    file = open(os.path.join(folder_results,"test_performances_fold{}.csv".format(nfold)), "w")
    file.write("Normalized CM : {}".format(np.round(cnf_matrix_norm, decimals=3).ravel()))
    file.write("\nFP\n"    )
    file.write(str(mismatched_fp))
    file.write("\nFN\n"    )
    file.write(str(mismatched_fn))
    file.write("\nCM : {}".format(cnf_matrix.ravel()))
    file.write("\nTest Acc\n"    )
    file.write(str(np.round(score[1],4)))
    file.close()

    print("\n\n  ****************  Completed FOLD ", nfold)
    #clear tf Session
    keras.backend.clear_session()
