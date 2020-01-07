# -*- coding: utf-8 -*-
"""
Authors: A. Chen, M. Balter, T. Maguire, M. Yarmush
Affiliation: Rutgers, The State University of New Jersey
Last updated: January 6, 2020

"""

import os
import dus_trainer
import tensorflow as tf

tf.reset_default_graph()

# Define model type
model_type = 'RFCN' # RFCN, FCN

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
model = dus_trainer.ModelTrainer(checkpointdirectory, modelfilename, logfilename, predfilename, model_type)

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
patchSize = (439, 584)
model.setTrainingPatchSize(patchSize[0], patchSize[1])
model.setValidationPatchSize(patchSize[0], patchSize[1])
model.setTestingPatchSize(patchSize[0], patchSize[1])

# %% Train model

tf.reset_default_graph()

model.train(model.training_list,
            None,
            None,
            model_type = model_type,
            model_checkpoint = None, 
            restartdirectoryname = checkpointdirectory + modelfilename + '_iter150.ckpt',
            restart_from = 150, # restart_from = -1
            epochs = 201,
            epochs_save = 2,
            learning_rate = 0.000002,
            loss_type = 'WCE',
            loss_weights = (0.8, 0.2, 0.0, 0.0, 0.0),
            pos_weight = 3.0,
            threshold = 0.5,
            k_composite = (0.5, 0.5, 0.5, 0.5, 0.5),
            min_max = (0, 1, 0, 1, 0, 1, 0, 1, 0, 1),
            n_augs = 1,
            batch_size = 8,
            batch_size_val = 8,
            window_size = 1,
            input_channels = 2,
            output_channels = 2,
            n_channels = 4,
            crop_from_seg_center = False,
            apply_augmentation = True,
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
           restartdirectoryname = checkpointdirectory + modelfilename + '_iter150.ckpt',
           restart_from = 150,
           loss_type = 'WCE',
           loss_weights = (1.0, 0.0, 0.0, 0.0, 0.0),
           pos_weight = 3.0,
           threshold = 0.5,
           k_composite = (0.5, 0.5, 0.5, 0.5, 0.5),
           min_max = (0, 1, 0, 1, 0, 1, 0, 1, 0, 1),
           n_augs = 1,
           batch_size = 8,
           window_size = 1,
           input_channels = 2,
           output_channels = 2,
           n_channels = 4,
           crop_from_seg_center = False,
           apply_augmentation = False,
           predict_from_label = True,
           write_prediction_text_file = False,
           write_images = False,
           show_images = True,
           show_activation_maps = False
           )
