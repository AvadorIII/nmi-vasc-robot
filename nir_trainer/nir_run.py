# -*- coding: utf-8 -*-
"""
Authors: A. Chen, M. Balter, T. Maguire, M. Yarmush
Affiliation: Rutgers, The State University of New Jersey
Last updated: January 6, 2020

"""

import os
import nir_trainer
import tensorflow as tf

tf.reset_default_graph()

# Define model type
model_type = 'RFCN' # FCN, RFCN

# Set paths
if model_type == 'FCN':
    modelfilename           = 'model_fcn'
    trainingfilename        = 'training_list_fcn.txt'
    validationfilename      = 'validation_list_fcn.txt'
    testingfilename         = 'testing_list_fcn.txt'
elif model_type == 'RFCN':
    modelfilename           = 'model_rfcn'
    trainingfilename        = 'training_list_rfcn.txt'
    validationfilename      = 'validation_list_rfcn.txt'
    testingfilename         = 'testing_list_rfcn.txt'
    
basedirectory               = os.getcwd()
checkpointdirectory         = basedirectory + '\\' + 'checkpoints' + '\\'
logfilename                 = checkpointdirectory + modelfilename + '_log.txt'
predfilename                = checkpointdirectory + modelfilename + '_pred.txt'

trainfilepath               = basedirectory + '\\' + 'datalists' + '\\' + trainingfilename
validationfilepath          = basedirectory + '\\' + 'datalists' + '\\' + validationfilename
testingfilepath             = basedirectory + '\\' + 'datalists' + '\\' + testingfilename

if not os.path.exists(checkpointdirectory):
    os.mkdir(checkpointdirectory)
    
# Constructor
model = nir_trainer.ModelTrainer(checkpointdirectory, modelfilename, logfilename, predfilename, model_type)

# Create data lists
if model_type == 'FCN':
    model.setTrainingData(trainfilepath)
    model.setValidationData(validationfilepath)
    model.setTestingData(testingfilepath)
elif model_type == 'RFCN':
    model.setRFCNTrainingData(trainfilepath)
    model.setRFCNValidationData(validationfilepath)
    model.setRFCNTestingData(testingfilepath)

# Define patch size
patchSize = (188, 142)
model.setTrainingPatchSize(patchSize[0], patchSize[1])
model.setValidationPatchSize(patchSize[0], patchSize[1])
model.setTestingPatchSize(patchSize[0], patchSize[1])

# Optionally, scale the values of the displacement field so that the MSE loss is approximately the same scale as the Dice loss
# If scaling is applied, then need to multiply back into the displacement field prediction before computing depth offset
displ_vmin = -17.1608 # scale by the average min v displacement across training set
displ_vmax  =  8.9163 # scale by the average max v displacement across training set
displ_umin = -38.6142 # scale by the average min u displacement across training set
displ_umax =  -5.2735 # scale by the average max u displacement across training set

# %% Train model

tf.reset_default_graph()

model.train(model.training_list,
            None,
            None,
            model_type = model_type,
            model_checkpoint = None, 
            restartdirectoryname = checkpointdirectory + modelfilename + '_iter3000.ckpt',
            restart_from = 3000, # restart_from = -1
            epochs = 4001,
            epochs_save = 5, # 2
            learning_rate = 0.000002,
            loss_type = 'WCE+Dice+MSE+L1',
            loss_weights = (0.1, 0.5, 0.3, 0.1),
            pos_weight = 4.0,
            threshold = 0.5,
            k_composite = (0.5, 0.5, 0.5, 0.5, 0.5),
            min_max = (0, 1, 0, 1, 0, 1, displ_vmin, displ_vmax, displ_umin, displ_umax),
            n_augs = 1,
            batch_size = 8,
            batch_size_val = 8,
            window_size = 1,
            input_channels = 2,
            output_channels = 3,
            n_channels = 4,
            crop_from_seg_center = False,
            apply_augmentation = False,
            shuffle_sequence_order = True,
            apply_reverse_order = True,
            predict_from_label = False,
            write_prediction_text_file = False,
            write_images = False,
            show_images = False,
            show_activation_maps = False
            )

# %% Evaluate trained model

model.test(model.testing_list,
           model_type = model_type,
           model_checkpoint = None, 
           restartdirectoryname = checkpointdirectory + modelfilename + '_iter3000.ckpt', # restartdirectoryname = None
           restart_from = 3000, # restart_from = -1
           loss_type = 'WCE+Dice+MSE+L1',
           loss_weights = (0.0025, 0.025, 2.5, 2.5, 0.0),
           pos_weight = 4.0,
           threshold = 0.5,
           k_composite = (0.5, 0.5, 0.5, 0.5, 0.5),
           min_max = (0, 1, 0, 1, 0, 1, displ_vmin, displ_vmax, displ_umin, displ_umax),
           n_augs = 1,
           batch_size = 1,
           window_size = 1,
           input_channels = 2,
           output_channels = 3,
           n_channels = 4,
           crop_from_seg_center = False,
           apply_augmentation = False,
           predict_from_label = True,
           write_prediction_text_file = False,
           write_images = False,
           show_images = True,
           show_activation_maps = False
           )
