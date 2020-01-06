# -*- coding: utf-8 -*-
"""
Authors: A. Chen, M. Balter, T. Maguire, M. Yarmush
Affiliation: Rutgers, The State University of New Jersey
Last updated: January 6, 2020

"""

#%% Import packages

import tensorflow as tf
import numpy as np
import datetime
import sys
import os
import random
import scipy.io
import SimpleITK as sitk

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline') # get_ipython().run_line_magic('matplotlib', 'qt')

sys.path.insert(1, '..\\fcn_model\\')
from fcn_utilities import countList2D, multireplace, showActivationMaps, showImagesAsSubplots, writeOutputsToFile
from fcn_lossfunc import sigmoid, computeDiceLoss_TF, evalLoss_customized
import fcn_model as model

# %% ModelTrainer
    
class ModelTrainer:
    def __init__(self, checkpointdirectoryname, checkpointprefix, logfilename, predfilename, model_type):
        self.checkpointdirectoryname = checkpointdirectoryname
        self.checkpointprefix = checkpointprefix
        self.logfilename = logfilename
        self.predfilename = predfilename
        self.model_type = model_type
        self.training_list = None
        self.validation_list = None
        self.testing_list = None
        
        
    def setTrainingData(self, filename):
        with open(filename) as file:
            files = file.readlines()
        for i in range(len(files)):
            files[i] = files[i].strip()
        self.training_list = [files.copy()]
    
    
    def setRFCNTrainingData(self, filename):
        with open(filename) as file:
            files = file.readlines()
        for i in range(len(files)):
            files[i] = files[i].strip()
        splits = [i for i, e in enumerate(files) if e == '-'] # array containing the row indices of '-' dash
        training_list = [None]*len(splits)
        startind = 0
        for i in range(len(splits)):
            #print('Training splits[i]: ' + str(splits[i]) + ', block length: ' + str(splits[i] - startind))
            if (splits[i] - startind) >= splits[0]: # only include elements that are full
                training_list[i] = files[startind:splits[i]]
            startind = splits[i]+1
        # Remove empty elements from the file list
        print('Number of training sequences (initial): ' + str(len(training_list)))
        training_list = list(filter(None, training_list))
        print('Number of training sequences (included): ' + str(len(training_list)))
        self.training_list = training_list.copy()
    
    
    def setTrainingPatchSize(self, xmax, ymax):
        self.training_xmax = xmax
        self.training_ymax = ymax
    
    
    def setValidationData(self, filename):
        with open(filename) as file:
            files = file.readlines()
        for i in range(len(files)):
            files[i] = files[i].strip()
        self.validation_list = [files.copy()]


    def setRFCNValidationData(self, filename):
        with open(filename) as file:
            files = file.readlines()
        for i in range(len(files)):
            files[i] = files[i].strip()
        splits = [i for i, e in enumerate(files) if e == '-'] # array containing the row indices of '-' dash
        validation_list = [None]*len(splits)
        startind = 0
        for i in range(len(splits)):
            #print('Validation splits[i]: ' + str(splits[i]) + ', block length: ' + str(splits[i] - startind))
            if (splits[i] - startind) >= splits[0]: # only include elements that are full
                validation_list[i] = files[startind:splits[i]]
            startind = splits[i]+1
        # Remove empty elements from the file list
        print('Number of validation sequences (initial): ' + str(len(validation_list)))
        validation_list = list(filter(None, validation_list)) 
        print('Number of validation sequences (included): ' + str(len(validation_list)))
        self.validation_list = validation_list.copy()
    
    
    def setValidationPatchSize(self, xmax, ymax):
        self.validation_xmax = xmax
        self.validation_ymax = ymax
        
        
    def setTestingPatchSize(self, xmax, ymax):
        self.test_xmax = xmax
        self.test_ymax = ymax
    
    
    def setTestingData(self, filename):
        with open(filename) as file:
            files = file.readlines()
        for i in range(len(files)):
            files[i] = files[i].strip()
        self.testing_list = [files.copy()]
    
    
    def setRFCNTestingData(self, filename):
        with open(filename) as file:
            files = file.readlines()
        for i in range(len(files)):
            files[i] = files[i].strip()
        splits = [i for i, e in enumerate(files) if e == '-'] # array containing the row indices of '-' dash
        testing_list = [None]*len(splits)
        startind = 0
        for i in range(len(splits)):
            #print('Testing splits[i]: ' + str(splits[i]) + ', block length: ' + str(splits[i] - startind))
            #if (splits[i] - startind) >= splits[0]:
            #   testing_list[i] = files[startind:splits[i]]
            testing_list[i] = files[startind:splits[i]] # in testing, can include all elements since not training with batches
            startind = splits[i]+1
        # Remove empty elements from the file list
        print('Number of testing sequences (initial): ' + str(len(testing_list)))
        testing_list = list(filter(None, testing_list)) 
        print('Number of testing sequences (included): ' + str(len(testing_list)))
        self.testing_list = testing_list.copy()
    
    
    def setRFCNFirstFrameData(self, filename):
        with open(filename) as file:
            lines = file.readlines()
            filelist = []
            anglelist = []
            for i in range(len(lines)):
                myline = lines[i].strip()
                ind = myline.rfind(' ')
                filelist.append(myline[0:ind])
                anglelist.append(myline[ind+1:len(myline)])
            self.first_frame_list = filelist.copy()
            self.first_frame_angle_list = anglelist.copy()
    
        
    def augmentData(self, filename, thetaz, thetay, scale, flipx, flipy, patch):
        return
    
    
    # %% Create training batch
    def createMiniBatch(self, file_list, batch_size, iter_batches, iter_timestep, compositetransforms, horizontalflip, crop_size, window_size, apply_augmentation=True, min_max=[0, 1, 0, 1, 0, 1, 0, 1, 0, 1]):
        
        start_time = datetime.datetime.now()
        
        mini_batch_x = [None]*batch_size
        mini_batch_y = [None]*batch_size
        mini_batch_filestr = [None]*batch_size 
        
        for iter_frame in range(batch_size):
            
            # Read data structure
            mat_filestr = file_list[iter_batches * batch_size + iter_frame][iter_timestep]
            mat_struct = scipy.io.loadmat(mat_filestr)
            
            # Get spline nodes structure
            splines = mat_struct.get('Label_info')
            
            # Get B-mode image
            img_arr = mat_struct.get('Image_bmode')
            img_arr = (img_arr - min_max[0] / (min_max[1] - min_max[0]))
            img = sitk.GetImageFromArray(img_arr)
            img_size = img.GetSize()
            img_center = [int(img_size[0]/2), int(img_size[1]/2)]
                    
            # Get doppler image
            doppler_arr_rgb = mat_struct.get('Image_doppler')
            doppler_arr = doppler_arr_rgb[:,:,2] - doppler_arr_rgb[:,:,0]
            doppler_arr = (doppler_arr - min_max[2] / (min_max[3] - min_max[2]))
            doppler = sitk.GetImageFromArray(doppler_arr)
                    
            # Get segmentation label (multi-class)
            seg_arr = mat_struct.get('Label_mask')
            seg_v_arr = np.zeros(seg_arr.shape)
            seg_a_arr = np.zeros(seg_arr.shape)
            for i in range(len(splines)):
                spline_i = splines[i][0][0][0]
                spline_i_class = spline_i[2][0]
                if spline_i_class == 'V':
                    seg_v_arr[np.where(seg_arr == (i+1))] = 1
                elif spline_i_class == 'A':
                    seg_a_arr[np.where(seg_arr == (i+1))] = 1
            seg_v_arr = (seg_v_arr - min_max[4] / (min_max[5] - min_max[4]))
            seg_a_arr = (seg_a_arr - min_max[4] / (min_max[5] - min_max[4]))
            seg_v = sitk.GetImageFromArray(seg_v_arr)
            seg_a = sitk.GetImageFromArray(seg_a_arr) 
            
            # Apply augmentation to the image, label, and doppler channels
            if apply_augmentation == True:
                outimg = sitk.Resample(img, 
                                       img.GetSize(), 
                                       compositetransforms[iter_frame], 
                                       sitk.sitkLinear, 
                                       (0, 0), 
                                       img.GetSpacing(), 
                                       (1, 0, 0, 1), 
                                       0.0, 
                                       sitk.sitkFloat32
                                       )
                
                outdoppler = sitk.Resample(doppler, 
                                           doppler.GetSize(), 
                                           compositetransforms[iter_frame], 
                                           sitk.sitkNearestNeighbor, 
                                           (0, 0), 
                                           doppler.GetSpacing(), 
                                           (1, 0, 0, 1), 
                                           0.0, 
                                           sitk.sitkFloat32
                                           )
                
                outseg_v = sitk.Resample(seg_v, 
                                         seg_v.GetSize(), 
                                         compositetransforms[iter_frame], 
                                         sitk.sitkNearestNeighbor, 
                                         (0, 0), 
                                         seg_v.GetSpacing(), 
                                         (1, 0, 0, 1), 
                                         0.0, 
                                         sitk.sitkFloat32
                                         )
                
                outseg_a = sitk.Resample(seg_a, 
                                         seg_a.GetSize(), 
                                         compositetransforms[iter_frame], 
                                         sitk.sitkNearestNeighbor, 
                                         (0, 0), 
                                         seg_a.GetSpacing(), 
                                         (1, 0, 0, 1), 
                                         0.0, 
                                         sitk.sitkFloat32
                                         )
            
                outimg_arr = sitk.GetArrayFromImage(outimg)
                outseg_v_arr = sitk.GetArrayFromImage(outseg_v)
                outseg_a_arr = sitk.GetArrayFromImage(outseg_a)
                outdoppler_arr = sitk.GetArrayFromImage(outdoppler)
                
            else:
                outimg_arr = img_arr
                outseg_v_arr = seg_v_arr
                outseg_a_arr = seg_a_arr
                outdoppler_arr = doppler_arr
                
            # Define cropping from center of transformed image
            img_center_arr = [img_center[1], img_center[0]] # flip order of numpy and sitk image axes
            crop_xmin = img_center_arr[0] - int(crop_size[0]/2)
            crop_xmax = img_center_arr[0] + int(crop_size[0]/2 + 0.5)
            crop_ymin = img_center_arr[1] - int(crop_size[1]/2)
            crop_ymax = img_center_arr[1] + int(crop_size[1]/2 + 0.5)
            
            # Apply the cropping
            outimg_crop = outimg_arr[crop_xmin:crop_xmax, crop_ymin:crop_ymax]
            outseg_v_crop = outseg_v_arr[crop_xmin:crop_xmax, crop_ymin:crop_ymax]
            outseg_a_crop = outseg_a_arr[crop_xmin:crop_xmax, crop_ymin:crop_ymax]
            outdoppler_crop = outdoppler_arr[crop_xmin:crop_xmax, crop_ymin:crop_ymax]
                    
            # Apply random horizontal flip
            if apply_augmentation == True and horizontalflip is not None:
                if horizontalflip[iter_frame] == 1:
                    outimg_crop = np.flip(outimg_crop, axis=1)
                    outseg_v_crop = np.flip(outseg_v_crop, axis=1)
                    outseg_a_crop = np.flip(outseg_a_crop, axis=1)
                    outdoppler_crop = np.flip(outdoppler_crop, axis=1)
            
            # Add to minibatch array
            mini_batch_x[iter_frame] = np.dstack((outimg_crop, outdoppler_crop)) # two-channel input
            mini_batch_y[iter_frame] = np.dstack((outseg_v_crop, outseg_a_crop)) # two-channel output
            mini_batch_filestr[iter_frame] = mat_filestr
            # print('iter_batches=' + str(iter_batches) + ', iter_timestep=' + str(iter_timestep) + ', iter_frame=' + str(iter_frame) + ', window_size=' + str(window_size) +
            #       ', mb_len=' + str(len(mini_batch_x)) + ', mb_size_x=' + str(mini_batch_x[iter_frame].shape) + ', mb_size_y=' + str(mini_batch_y[iter_frame].shape)) 
                                   
        # Finally, reshape mini_batch so the batch dimension is first
        mini_batch_x = np.reshape(np.stack(mini_batch_x, axis=0), [batch_size, mini_batch_x[0].shape[0], mini_batch_x[0].shape[1], mini_batch_x[0].shape[2], window_size])
        mini_batch_y = np.reshape(np.stack(mini_batch_y, axis=0), [batch_size, mini_batch_y[0].shape[0], mini_batch_y[0].shape[1], mini_batch_y[0].shape[2], window_size])
        
        end_time = datetime.datetime.now()
        time_elapsed = end_time - start_time
        # print('Time (create minibatch), ms=' + str(time_elapsed.total_seconds() * 1000))
        
        return mini_batch_x, mini_batch_y, mini_batch_filestr
    
    
    # %% Create testing batch
    def createMiniBatchForTesting(self, file_list, batch_size, iter_batches, iter_timestep, compositetransforms, horizontalflip, crop_size, window_size, apply_augmentation=True, min_max=[0, 1, 0, 1]):
        
        start_time = datetime.datetime.now()
    
        mini_batch_x = [None]*batch_size
        mini_batch_filestr = [None]*batch_size
        
        for iter_frame in range(batch_size):
            
            # Read data structure
            mat_filestr = file_list[iter_batches * batch_size + iter_frame][iter_timestep]
            mat_struct = scipy.io.loadmat(mat_filestr)
                           
            # Get B-mode image
            img_arr = mat_struct.get('Image_bmode')
            img_arr = (img_arr - min_max[0] / (min_max[1] - min_max[0]))
            img = sitk.GetImageFromArray(img_arr)
            img_size = img.GetSize()
            img_center = [int(img_size[0]/2), int(img_size[1]/2)]
                    
            # Get doppler image
            doppler_arr_rgb = mat_struct.get('Image_doppler')
            doppler_arr = doppler_arr_rgb[:,:,2] - doppler_arr_rgb[:,:,0]
            doppler_arr = (doppler_arr - min_max[2] / (min_max[3] - min_max[2]))
            doppler = sitk.GetImageFromArray(doppler_arr)
            
            # Apply augmentation to the image, label, and doppler channels
            if apply_augmentation == True:
                outimg = sitk.Resample(img, 
                                       img.GetSize(), 
                                       compositetransforms[iter_frame], 
                                       sitk.sitkLinear, 
                                       (0, 0), 
                                       img.GetSpacing(), 
                                       (1, 0, 0, 1), 
                                       0.0, 
                                       sitk.sitkFloat32
                                       )
                
                outdoppler = sitk.Resample(doppler, 
                                           doppler.GetSize(), 
                                           compositetransforms[iter_frame], 
                                           sitk.sitkNearestNeighbor, 
                                           (0, 0), 
                                           doppler.GetSpacing(), 
                                           (1, 0, 0, 1), 
                                           0.0, 
                                           sitk.sitkFloat32
                                           )
                
                outimg_arr = sitk.GetArrayFromImage(outimg)
                outdoppler_arr = sitk.GetArrayFromImage(outdoppler)
                
            else:
                outimg_arr = img_arr
                outdoppler_arr = doppler_arr
            
            # Define cropping from center of transformed image
            img_center_arr = [img_center[1], img_center[0]] # flip order of numpy and sitk image axes
            crop_xmin = img_center_arr[0] - int(crop_size[0]/2)
            crop_xmax = img_center_arr[0] + int(crop_size[0]/2 + 0.5)
            crop_ymin = img_center_arr[1] - int(crop_size[1]/2)
            crop_ymax = img_center_arr[1] + int(crop_size[1]/2 + 0.5)
            
            # Apply the cropping
            outimg_crop = outimg_arr[crop_xmin:crop_xmax, crop_ymin:crop_ymax]
            outdoppler_crop = outdoppler_arr[crop_xmin:crop_xmax, crop_ymin:crop_ymax]
                    
            # Apply random horizontal flip
            if apply_augmentation == True and horizontalflip is not None:
                if horizontalflip[iter_frame] == 1:
                    outimg_crop = np.flip(outimg_crop, axis=1)
                    outdoppler_crop = np.flip(outdoppler_crop, axis=1)                            
                    
            # Add to minibatch array
            mini_batch_x[iter_frame] = np.dstack((outimg_crop, outdoppler_crop)) # two-channel input
            mini_batch_filestr[iter_frame] = mat_filestr
                                   
        # Finally, reshape mini_batch so the batch dimension is first
        mini_batch_x = np.reshape(np.stack(mini_batch_x, axis=0), [batch_size, mini_batch_x[0].shape[0], mini_batch_x[0].shape[1], mini_batch_x[0].shape[2], window_size])
                                    
        end_time = datetime.datetime.now()
        time_elapsed = end_time - start_time
        # print('Time (create minibatch for testing), ms=' + str(time_elapsed.total_seconds() * 1000))
        return mini_batch_x, mini_batch_filestr
    
    
    # %% Compute composite transforms
    def computeCompositeTransforms(self, file_list, batch_size, iter_batches, crop_size, augmentation_params, crop_from_seg_center=False):
        
        start_time = datetime.datetime.now()
    
        (k_transx, k_transy, k_rot, k_scale, k_flip) = augmentation_params
        
        seg_centers = [None] * batch_size
        compositetransforms = [None] * batch_size
        horizontalflip = [None] * batch_size
        
        for iter_frame in range(batch_size):
                
            # Read data structure
            mat_filestr = file_list[iter_batches * batch_size + iter_frame][0]
            mat_struct = scipy.io.loadmat(mat_filestr)
            
            # Get spline nodes structure
            splines = mat_struct.get('Label_info')
            
            # Get left image
            img_arr = mat_struct.get('Image_bmode')
            img = sitk.GetImageFromArray(img_arr)
            img_size = img.GetSize()
            img_center = [int(img_size[0]/2), int(img_size[1]/2)]
            
            # Get segmentation label (multi-class)
            if crop_from_seg_center == True:
                seg_stats = []
                for i in range(len(splines)):
                    spline_i = splines[i][0][0][0]
                    spline_i_ctr = spline_i[1][0]
                    seg_stats.append([float(spline_i_ctr[0]), float(spline_i_ctr[1])])
                
                # If multiple segmentation objects, randomly pick one
                seg_rand_idx = random.randint(0, len(seg_stats)-1)
                            
                # Store segmentation centers in array
                seg_centers[iter_frame] = [seg_stats[seg_rand_idx][0], seg_stats[seg_rand_idx][1]]            
    
            else:  
            
                # Store segmentation centers in array
                seg_centers[iter_frame] = [float(img_center[0]), float(img_center[1])]
                
                
            # First center the transformation on the segmentation centroid
            translate_x = seg_centers[iter_frame][0] - img_center[0]
            translate_y = seg_centers[iter_frame][1] - img_center[1]
                    
            translate_x += crop_size[0] * k_transx * (np.random.rand(1)[0] - 0.5)
            translate_y += crop_size[1] * k_transy * (np.random.rand(1)[0] - 0.5)
            rotate_radn  = np.pi        * k_rot    * (np.random.rand(1)[0] - 0.5)
            scaling      = 1.0          + k_scale  * (np.random.rand(1)[0] - 0.5)
                    
            # Rotation transform
            rotationTransform = sitk.Euler2DTransform()
            rotationTransform.SetAngle(rotate_radn)
            rotationTransform.SetCenter(seg_centers[iter_frame])
            
            # Translation transform
            translationTransform = sitk.TranslationTransform(2, (translate_x, translate_y))
                    
            # Scale transform
            scaleTransform = sitk.ScaleTransform(2)
            scaleTransform.SetScale((scaling, scaling))
            scaleTransform.SetCenter(seg_centers[iter_frame])
            
            # Final composite transform
            compositetransforms[iter_frame] = sitk.Transform(2, sitk.sitkComposite)
            compositetransforms[iter_frame].AddTransform(rotationTransform)
            compositetransforms[iter_frame].AddTransform(translationTransform)
            compositetransforms[iter_frame].AddTransform(scaleTransform)
            
            # Random horizontal flip
            if np.random.rand(1) < k_flip:
                horizontalflip[iter_frame] = 1
            else:
                horizontalflip[iter_frame] = 0
                
        end_time = datetime.datetime.now()
        time_elapsed = end_time - start_time
        # print('Time (composite transforms), ms=' + str(time_elapsed.total_seconds() * 1000))
        
        return compositetransforms, horizontalflip
    
    
    # %% Set and evaluate loss functions
    def setLoss_TF(self, logits, labels, loss_type='WCE', loss_weights=(1.0, 1.0, 1.0, 0.0, 0.0), pos_weight=1.0):
    
        if loss_type == 'WCE':
            loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=labels, pos_weight=pos_weight))
        
        elif loss_type == 'Dice':
            loss = computeDiceLoss_TF(logits, labels) # loss = tf.reduce_mean(computeGeneralizedDiceLoss_TF(logits, labels, dim=3))
            
        if loss_type == 'MSE':
            loss = tf.reduce_mean(tf.square(logits - labels))
            
        elif loss_type == 'WCE+Dice':
            loss_wce = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=labels, pos_weight=pos_weight))
            loss_dice = computeDiceLoss_TF(logits, labels) # loss_dice = tf.reduce_mean(computeGeneralizedDiceLoss_TF(logits, labels, dim=3))
            loss = loss_weights[0] * loss_wce + loss_weights[1] * loss_dice
            
        elif loss_type == 'WCE+MSE':
            loss_wce = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=labels, pos_weight=pos_weight)) # apply to channels 0 and 1
            loss_mse = tf.reduce_mean(tf.square(logits - labels)) # apply to channel 2
            loss = loss_weights[1] * loss_wce + loss_weights[2] * loss_mse
        
        elif loss_type == 'WCE+Dice+MSE':
            loss_wce = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=labels, pos_weight=pos_weight)) # apply to channels 0 and 1
            loss_dice = computeDiceLoss_TF(logits, labels) # apply to channels 0 and 1
            loss_mse = tf.reduce_mean(tf.square(logits - labels)) # apply to channel 2
            loss = loss_weights[0] * loss_wce + loss_weights[1] * loss_dice + loss_weights[2] * loss_mse
        
        elif loss_type == 'WCE+Dice+MSE+L1':
            loss_wce = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=labels, pos_weight=pos_weight)) # apply to channels 0 and 1
            loss_dice = computeDiceLoss_TF(logits, labels) # apply to channels 0 and 1
            loss_mse = tf.reduce_mean(tf.square(logits - labels)) # apply to channel 2
            loss_tv = tf.reduce_sum(tf.image.total_variation(logits)) / tf.cast(tf.size(logits), dtype=tf.float32) # apply to channel 2 (total variation of displacement field)
            loss = loss_weights[0] * loss_wce + loss_weights[1] * loss_dice + loss_weights[2] * loss_mse + loss_weights[3] * loss_tv
        
        elif loss_type == 'WCE+Dice+MSE+L2':
            loss_wce = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=labels, pos_weight=pos_weight)) # apply to channels 0 and 1
            loss_dice = computeDiceLoss_TF(logits, labels) # apply to channels 0 and 1
            loss_mse = tf.reduce_mean(tf.square(logits - labels)) # apply to channel 2
            (grady, gradx) = tf.image.image_gradients(logits) # apply to channel 2 (gradient of output tensor in x and y)
            loss_grad = tf.reduce_mean(tf.square(grady)) + tf.reduce_mean(tf.square(gradx)) # L2 norm of gradients
            loss = loss_weights[0] * loss_wce + loss_weights[1] * loss_dice + loss_weights[2] * loss_mse + loss_weights[4] * loss_grad
         
        return loss
    
    
    def evalLoss_TF(self, logits, labels, loss_type='WCE', loss_weights=(1.0, 1.0, 1.0, 0.0, 0.0), pos_weight=1.0):
        
        # For classification loss
        logits = logits.astype(np.float32)
        labels = labels.astype(np.float32)
        
        # Display raw value parameters
        print('logits (shape)=' + str(logits.shape) + ', labels (shape)=' + str(logits.shape))
        print('logits (min, max)=' + str(np.amin(logits)) + ', ' + str(np.amax(logits)) + '; labels (min, max)=' + str(np.amin(labels)) + ', ' + str(np.amax(labels)))
        print('logits (reduce_mean)=' + str(np.mean(logits)) + '; labels (reduce_mean)=' + str(np.mean(labels)))
       
        loss_wce = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=labels, pos_weight=pos_weight))
        loss_dice = computeDiceLoss_TF(logits, labels)
        loss_mse = tf.reduce_mean(tf.square(logits - labels))
         
        print('Loss type: ' + str(loss_type) + ', loss_weights=' + str(loss_weights))
        if loss_type == 'WCE':
            loss = loss_weights[0] * loss_wce
        elif loss_type == 'Dice':
            loss = loss_weights[1] * loss_dice
        elif loss_type == 'MSE':
            loss = loss_weights[2] * loss_mse
        elif loss_type == 'WCE+Dice':
            loss = loss_weights[0] * loss_wce + loss_weights[1] * loss_dice
        elif loss_type == 'WCE+MSE':
            loss = loss_weights[0] * loss_wce + loss_weights[2] * loss_mse
        elif loss_type == 'WCE+Dice+MSE':
            loss = loss_weights[0] * loss_wce + loss_weights[1] * loss_dice + loss_weights[2] * loss_mse
        else:
            loss = loss_weights[0] * loss_wce # default to WCE
        
        loss_wce  = loss_wce.eval()
        loss_dice = loss_dice.eval()
        loss_mse  = loss_mse.eval()
        loss_tv   = 0.0
        loss_grad = 0.0
        loss      = loss.eval()
    
        print('loss_wce='  + str(loss_wce)  + ', weight=' + str(loss_weights[0]) + ', loss_wce_weight='  + str(loss_weights[0] * loss_wce))
        print('loss_dice=' + str(loss_dice) + ', weight=' + str(loss_weights[1]) + ', loss_dice_weight=' + str(loss_weights[1] * loss_dice))
        print('loss_mse='  + str(loss_mse)  + ', weight=' + str(loss_weights[2]) + ', loss_mse_weight='  + str(loss_weights[2] * loss_mse))
        print('loss_tv='   + str(loss_tv)   + ', weight=' + str(loss_weights[3]) + ', loss_tv_weight='   + str(loss_weights[3] * loss_tv))
        print('loss_grad=' + str(loss_grad) + ', weight=' + str(loss_weights[4]) + ', loss_grad_weight=' + str(loss_weights[4] * loss_grad))
        print('loss='      + str(loss))
            
        return (loss_wce, loss_dice, loss_mse, loss_tv, loss_grad, loss)
    
        
    def evalLoss_multichannel(self, X, Y, pos_weight, threshold, iter_batch, calc_TF=False, calc_regularizers=False, calc_array=[True, True, True, True, True, True]):
        start_time = datetime.datetime.now()
        print('\n')
        
        calcWeightedCrossEntropy = [0, 0, 0, 0, 0, 0]
        calcLossDiceSoft = [0, 0, 0, 0, 0, 0]
        calcLossDiceHard = [0, 0, 0, 0, 0, 0]
        calcMeanSquares = [0, 0, 0, 0, 0, 0]
        
        # Veins
        i = 0
        if calc_array[i] == True:
            print('>>>> Input channel 1:')
            logits = X[iter_batch,:,:,0,0].astype(np.float32)
            labels = Y[iter_batch,:,:,0,0].astype(np.float32)
            (calcWeightedCrossEntropy[i], calcLossDiceSoft[i], calcLossDiceHard[i], calcMeanSquares[i], _, _) = evalLoss_customized(logits, labels, pos_weight, threshold, calc_TF, calc_regularizers)
            print('\n')
            
        # Arteries
        i = 1
        if calc_array[i] == True:
            print('>>>> Input channel 2:')
            logits = X[iter_batch,:,:,1,0].astype(np.float32)
            labels = Y[iter_batch,:,:,1,0].astype(np.float32)
            (calcWeightedCrossEntropy[i], calcLossDiceSoft[i], calcLossDiceHard[i], calcMeanSquares[i], _, _) = evalLoss_customized(logits, labels, pos_weight, threshold, calc_TF, calc_regularizers)
            print('\n')
            
        # Channels
        i = 2
        if calc_array[i] == True:
            print('>>>> All channels:')
            logits = X[iter_batch,:,:,:,0].astype(np.float32)
            labels = Y[iter_batch,:,:,:,0].astype(np.float32)
            (calcWeightedCrossEntropy[i], calcLossDiceSoft[i], calcLossDiceHard[i], calcMeanSquares[i], _, _) = evalLoss_customized(logits, labels, pos_weight, threshold, calc_TF, calc_regularizers)
            print('\n')
                                                
        # Tensor
        i = 3
        if calc_array[i] == True:
            print('>>>> Full tensor:')
            logits = X[iter_batch,:,:,:,:].astype(np.float32)
            labels = Y[iter_batch,:,:,:,:].astype(np.float32)
            (calcWeightedCrossEntropy[i], calcLossDiceSoft[i], calcLossDiceHard[i], calcMeanSquares[i], _, _) = evalLoss_customized(logits, labels, pos_weight, threshold, calc_TF, calc_regularizers)
            print('\n')
                                               
        # Batch
        i = 4
        if calc_array[i] == True:
            print('>>>> Full batch:')
            logits = X[:,:,:,:,:].astype(np.float32)
            labels = Y[:,:,:,:,:].astype(np.float32)
            (calcWeightedCrossEntropy[i], calcLossDiceSoft[i], calcLossDiceHard[i], calcMeanSquares[i], _, _) = evalLoss_customized(logits, labels, pos_weight, threshold, calc_TF, calc_regularizers)
            print('\n')
            
        end_time = datetime.datetime.now()
        time_elapsed = end_time - start_time
        print('Time (compare predictions, multi-channel), ms=' + str(time_elapsed.total_seconds() * 1000))
        return (calcWeightedCrossEntropy, calcLossDiceSoft, calcLossDiceHard, calcMeanSquares)
    
    
    # %% Get results
    def getResults(self, mini_batch_x, mini_batch_y, mini_batch_pred, mini_batch_sigm, mini_batch_binary, 
                   mini_batch_filestr, batch_size, pos_weight, threshold, loss_type, loss_weights, myloss=0.0, time_elapsed=0.0, mylast_feature_list=None,
                   predict_from_label=False, write_prediction_text_file=False, write_images=False, show_images=False, show_activation_maps=False):
            
        # Loop through each image in current batch
        for iter_plot in range(batch_size):
            
            # Create write file paths 
            if write_images == True:
                filestr_split = mini_batch_filestr[iter_plot].split('\\')
                img_name = multireplace(filestr_split[-1], {'data_' : 'predictions_'})
                out_filestr = multireplace(mini_batch_filestr[iter_plot], {filestr_split[-2] + '\\' + filestr_split[-1] : 'output\\' + img_name})
                if not os.path.exists(out_filestr):
                    os.mkdir(out_filestr)
                out_filestr_subplots = out_filestr + '\\' + img_name + '_subplots.png'
               
                # Write image files and/or raw data
                writeOutputsToFile(out_filestr,
                                       img_name,
                                       iter_plot,
                                       mini_batch_x,
                                       mini_batch_y,
                                       mini_batch_pred,
                                       mini_batch_sigm,
                                       mini_batch_binary,
                                       write_raw_data=True
                                       )
            else:
                out_filestr_subplots = None
            
            # Show images and write prediction results
            if predict_from_label == True:
                print(mini_batch_filestr[iter_plot])
                if show_images == True:
                    showImagesAsSubplots([mini_batch_x[iter_plot,:,:,0,0],    mini_batch_x[iter_plot,:,:,1,0],
                                          mini_batch_y[iter_plot,:,:,0,0],    mini_batch_y[iter_plot,:,:,1,0],
                                          mini_batch_pred[iter_plot,:,:,0,0], mini_batch_pred[iter_plot,:,:,1,0],
                                          mini_batch_sigm[iter_plot,:,:,0,0], mini_batch_sigm[iter_plot,:,:,1,0]],
                                         ['Image left', 'Image right',
                                          'Label left', 'Label right', 
                                          'Prediction left', 'Prediction right',
                                          'Sigmoid left', 'Sigmoid right'],
                                         _figsize=(8, 6), n_rows=2, min_max=(0,1), _grid_yn=True, filestr=out_filestr_subplots)
                    
                # Calculate segmentation losses
                (loss_wce, loss_dice, loss_mse, loss_tv, loss_grad, loss) = self.evalLoss_TF(mini_batch_pred,
                                                                                             mini_batch_y,
                                                                                             loss_type,
                                                                                             loss_weights,
                                                                                             pos_weight
                                                                                             )
                
                (calcWeightedCrossEntropy, calcLossDiceSoft, calcLossDiceHard, calcMeanSquares) = self.evalLoss_multichannel(mini_batch_pred,
                                                                                                                             mini_batch_y,
                                                                                                                             pos_weight,
                                                                                                                             threshold,
                                                                                                                             iter_plot,
                                                                                                                             calc_TF=False,
                                                                                                                             calc_regularizers=False,
                                                                                                                             calc_array=[True, True, True, False, False, False]
                                                                                                                             )
                                
                if write_prediction_text_file == True:
                    pred_str = multireplace(mini_batch_filestr[iter_plot], {'.mat' : ''})
                    pred_str = pred_str + ': time(ms)='       + str('{:4.2f}'.format(time_elapsed.total_seconds() * 1000 / batch_size))
                    pred_str = pred_str + ', V_WCE = '        + str('{:6.4f}'.format(calcWeightedCrossEntropy[0]))
                    pred_str = pred_str + ', V_DiceSoft = '   + str('{:6.4f}'.format(calcLossDiceSoft[0]))
                    pred_str = pred_str + ', V_DiceHard = '   + str('{:6.4f}'.format(calcLossDiceHard[0]))
                    pred_str = pred_str + ', A_WCE = '        + str('{:6.4f}'.format(calcWeightedCrossEntropy[1]))
                    pred_str = pred_str + ', A_DiceSoft = '   + str('{:6.4f}'.format(calcLossDiceSoft[1]))
                    pred_str = pred_str + ', A_DiceHard = '   + str('{:6.4f}'.format(calcLossDiceHard[1]))
                    pred_str = pred_str + ', Loss (model) = ' + str('{:6.4f}'.format(myloss))
                    print('Writing to text file: ' + self.predfilename)
                    with open(self.predfilename, "a") as mypred:
                        mypred.write(pred_str + '\n')
            
            else:
                showImagesAsSubplots([mini_batch_x[iter_plot,:,:,0,0], mini_batch_x[iter_plot,:,:,1,0],
                                      mini_batch_y[iter_plot,:,:,0,0], mini_batch_y[iter_plot,:,:,1,0]],
                                     ['Image left', 'Image right', 'Empty',
                                      'Label left', 'Label right', 'Label displ'],
                                      _figsize=(8, 3), n_rows=1, min_max=(0,1), _grid_yn=True, filestr=out_filestr_subplots
                                      )
            
        # Show activation maps
        if show_activation_maps == True:
            showActivationMaps(mylast_feature_list, 4, True)
            
        return
    
    # %% Train model
    def train(self, training_list,
                    validation_list,
                    testing_list,
                    model_type = 'RFCN',
                    model_checkpoint = None, 
                    restartdirectoryname = None,
                    restart_from = -1,
                    epochs = 101,
                    epochs_save = 5,
                    learning_rate = 0.00001,
                    loss_type='WCE+Dice+MSE',
                    loss_weights = (1.0, 0.0, 0.0, 0.0, 0.0),
                    pos_weight = 3.0,
                    threshold = 0.5,
                    k_composite=(0.5, 0.5, 0.5, 0.5, 0.5),
                    min_max=(0, 1, 0, 1, 0, 1, 0, 1, 0, 1),
                    n_augs = 1,
                    batch_size = 8,
                    batch_size_val = 8,
                    window_size = 1,
                    input_channels = 2,
                    output_channels = 2,
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
                    ):
        
        if not os.path.isdir(self.checkpointdirectoryname):
            os.mkdir(self.checkpointdirectoryname)
            
        # Display some settings
        print('predict_from_label:          ' + str(predict_from_label))
        print('write_prediction_text_file:  ' + str(write_images))
        print('write_images:                ' + str(write_images))
        print('show_images:                 ' + str(write_prediction_text_file))
        print('show_activation_maps:        ' + str(show_activation_maps))
        print('input_channels:              ' + str(input_channels))
        print('output_channels:             ' + str(output_channels))
        print('n_channels:                  ' + str(n_channels))
        
        # Define crop size and center
        xsize = self.training_xmax
        ysize = self.training_ymax
        crop_size = [xsize, ysize]
        
        xsize_val = self.validation_xmax
        ysize_val = self.validation_ymax
        crop_size_val = [xsize_val, ysize_val]
        
        # Define training and validation model placeholders
        print('input_channels: ' + str(input_channels))
        print('output_channels: ' + str(input_channels))
        print('n_channels: ' + str(input_channels))
        with tf.name_scope('inputs'):
            x_input = tf.placeholder(tf.float32, shape=[batch_size, self.training_xmax, self.training_ymax, input_channels, window_size], name='x_input')
            y_input = tf.placeholder(tf.float32, shape=[batch_size, self.training_xmax, self.training_ymax, output_channels, window_size], name='y_input')
            c1_input = tf.placeholder(tf.float32, shape=[batch_size, xsize, ysize, n_channels], name='c1_input')
            c2_input = tf.placeholder(tf.float32, shape=[batch_size, int(np.ceil(xsize/2)), int(np.ceil(ysize/2)), n_channels*2], name='c2_input')
            c3_input = tf.placeholder(tf.float32, shape=[batch_size, int(np.ceil(xsize/4)), int(np.ceil(ysize/4)), n_channels*4], name='c3_input')
            c5_input = tf.placeholder(tf.float32, shape=[batch_size, int(np.ceil(xsize/8)), int(np.ceil(ysize/8)), n_channels*8], name='c5_input')
        
        with tf.name_scope('inputs'):
            x_input_val = tf.placeholder(tf.float32, shape=[batch_size_val, self.validation_xmax, self.validation_ymax, input_channels, window_size], name='x_input_val')
            y_input_val = tf.placeholder(tf.float32, shape=[batch_size_val, self.validation_xmax, self.validation_ymax, output_channels, window_size], name='y_input_val')
            c1_input_val = tf.placeholder(tf.float32, shape=[batch_size_val, xsize_val, ysize_val, n_channels], name='c1_input_val')
            c2_input_val = tf.placeholder(tf.float32, shape=[batch_size_val, int(np.ceil(xsize_val/2)), int(np.ceil(ysize_val/2)), n_channels*2], name='c2_input_val')
            c3_input_val = tf.placeholder(tf.float32, shape=[batch_size_val, int(np.ceil(xsize_val/4)), int(np.ceil(ysize_val/4)), n_channels*4], name='c3_input_val')
            c5_input_val = tf.placeholder(tf.float32, shape=[batch_size_val, int(np.ceil(xsize_val/8)), int(np.ceil(ysize_val/8)), n_channels*8], name='c5_input_val')
        
        # Build model graph
        print('>>>>>>>> Build training model graph:')
        if model_type == 'FCN':
            prediction, last_feature_list = model.fcn_outer_model_batchnorm(x_input, 
                                                                            c1_input, 
                                                                            c2_input, 
                                                                            c3_input, 
                                                                            c5_input, 
                                                                            input_channels=input_channels,
                                                                            output_channels=output_channels,
                                                                            n_channels=n_channels,
                                                                            training=True
                                                                            )

            # Build validation model graph
            print('>>>>>>>> Build validation model graph:')
            prediction_val, last_feature_list_val = model.fcn_outer_batchnorm(x_input_val, 
                                                                               c1_input_val, 
                                                                               c2_input_val, 
                                                                               c3_input_val, 
                                                                               c5_input_val, 
                                                                               input_channels=input_channels,
                                                                               output_channels=output_channels,
                                                                               n_channels=n_channels, 
                                                                               reuse=tf.AUTO_REUSE,
                                                                               training=True
                                                                               )
            
        elif model_type == 'RFCN':
            prediction, last_feature_list = model.rfcn_outer_batchnorm(x_input, 
                                                                       c1_input, 
                                                                       c2_input, 
                                                                       c3_input, 
                                                                       c5_input, 
                                                                       input_channels=input_channels,
                                                                       output_channels=output_channels,
                                                                       n_channels=n_channels,
                                                                       training=True
                                                                       )
            
            # Build validation model graph
            print('>>>>>>>> Build validation model graph:')
            prediction_val, last_feature_list_val = model.rfcn_outer_batchnorm(x_input_val, 
                                                                               c1_input_val, 
                                                                               c2_input_val, 
                                                                               c3_input_val, 
                                                                               c5_input_val, 
                                                                               input_channels=input_channels,
                                                                               output_channels=output_channels,
                                                                               n_channels=n_channels, 
                                                                               reuse=tf.AUTO_REUSE,
                                                                               training=True
                                                                               )
           
        
        # Write out the model layer sizes
        prediction = tf.stack(prediction, axis=4) # tf.concat(prediction, axis=4)
        prediction_val = tf.stack(prediction_val, axis=4) # tf.concat(prediction_val, axis=4)
        print('prediction (after stack): ' + str(prediction.shape))
        print('prediction_val (after stack): ' + str(prediction_val.shape))
                
        # Define loss for training and validation (different due to image patch size, hardcoded)
        print('Loss function: ' + str(loss_type))
        train_loss = self.setLoss_TF(logits=prediction, labels=y_input, loss_type=loss_type, loss_weights=loss_weights, pos_weight=pos_weight)
        val_loss = self.setLoss_TF(logits=prediction_val, labels=y_input_val, loss_type=loss_type, loss_weights=loss_weights, pos_weight=pos_weight)
        
        # Get number of sequences, timesteps per sequence, batch iterations per epoch, and total number of evaluations
        n_sequences_train = len(training_list)
        n_timesteps_train = len(training_list[0])
        n_batches_train = int(n_sequences_train/batch_size)
        n_evals_train = countList2D(training_list)-(window_size-1)*len(training_list)
        
        if validation_list is not None:
            n_sequences_val = len(validation_list)
            n_timesteps_val = len(validation_list[0])
            n_batches_val = int(n_sequences_val/batch_size_val)
            n_evals_val = countList2D(validation_list)-(window_size-1)*len(validation_list)
        else:
            n_sequences_val = 0
            n_timesteps_val = 0
            n_batches_val = 0
            n_evals_val = 0
        
        if testing_list is not None:
            n_sequences_test = len(testing_list)
            n_timesteps_test = len(testing_list[0])
            n_batches_test = int(len(testing_list)/batch_size_val)
            n_evals_test = countList2D(testing_list)-(window_size-1)*len(testing_list)
        else:
            n_sequences_test = 0
            n_timesteps_test = 0
            n_batches_test = 0
            n_evals_test = 0
            
        print('n_sequences_train=' + str(n_sequences_train) + ', n_sequences_val=' + str(n_sequences_val) + ', n_sequences_test=' + str(n_sequences_test))
        print('n_timesteps_train=' + str(n_timesteps_train) + ', n_timesteps_val=' + str(n_timesteps_val) + ', n_timesteps_test=' + str(n_timesteps_test))
        print('n_batches_train='   + str(n_batches_train)   + ', n_batches_val=' + str(n_batches_val)     + ', n_batches_test='   + str(n_batches_test))
        print('n_evals_train='     + str(n_evals_train)     + ', n_evals_val=' + str(n_evals_val)         + ', n_evals_test='     + str(n_evals_test))
        
        # Set optimizers
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.995)
        opt_operation = opt.minimize(train_loss)
        
        # Build variables lists
        vnet_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "contracting_path") + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "expanding_path")
        vnet_saver = tf.train.Saver(vnet_vars) # Model loader
        saver = tf.train.Saver() # RFCN saver
        
        ########################################### Start training session ###########################################  
        config = tf.ConfigProto(device_count={'GPU':1})
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            if model_checkpoint is not None:
                vnet_saver.restore(sess, model_checkpoint)
            
            if restart_from > -1:
                if restartdirectoryname is None:
                    saver.restore(sess, str(self.checkpointdirectoryname) + '/' + str(self.checkpointprefix) + "_iter" + str(restart_from) + ".ckpt")
                else:
                    saver.restore(sess, restartdirectoryname)
            
            ######### Loop through all epochs #########
            for epoch in range(restart_from+1, epochs):
                start_time_epoch = datetime.datetime.now()
                
                epoch_loss_train = 0
                epoch_loss_val = 0                
                epoch_loss_test = 0
                    
                ######### Evaluate training loss #########
                print('\nEvaluate training loss (epoch ' + str(epoch) + '):')
                
                # Randomize data order
                training_list_copy = training_list.copy()
                if shuffle_sequence_order == True:
                    np.random.shuffle(training_list_copy)
                
                for iter_augs in range(n_augs):
                    for iter_batches in range(n_batches_train):
                        
                        # Compute composite transforms with data augmentation
                        compositetransforms, horizontalflip = self.computeCompositeTransforms(training_list_copy,
                                                                                              batch_size,
                                                                                              iter_batches,
                                                                                              crop_size,
                                                                                              k_composite,
                                                                                              crop_from_seg_center
                                                                                              )
                        
                        # Loop through each time step
                        for iter_timestep in range(len(training_list_copy[iter_batches*batch_size]) - window_size + 1):
                            
                            # Alternate between forward and backward sequence order during training
                            iter_timestep_apply = iter_timestep
                            if apply_reverse_order == True:
                                if (epoch % 2) != 0:
                                    iter_timestep_apply = n_timesteps_train - (iter_timestep + 1)
                            
                            # Create mini batch for training
                            mini_batch_x, mini_batch_y, mini_batch_filestr = self.createMiniBatch(training_list_copy, 
                                                                                                  batch_size,
                                                                                                  iter_batches,
                                                                                                  iter_timestep_apply,
                                                                                                  compositetransforms,
                                                                                                  horizontalflip,
                                                                                                  crop_size,
                                                                                                  window_size, 
                                                                                                  apply_augmentation,
                                                                                                  min_max
                                                                                                  )

                            # Initialize convolution layers for each time step
                            if iter_timestep == 0:
                                c1 = np.zeros(c1_input.get_shape())
                                c2 = np.zeros(c2_input.get_shape())
                                c3 = np.zeros(c3_input.get_shape())
                                c5 = np.zeros(c5_input.get_shape())
                            else:
                                c1 = mylast_feature_list[0]
                                c2 = mylast_feature_list[1]
                                c3 = mylast_feature_list[2]
                                c5 = mylast_feature_list[3]
                                                    
                            # Train model
                            _optimizer, myloss, mylast_feature_list = sess.run([opt_operation, train_loss, last_feature_list], 
                                                                                 feed_dict={x_input: mini_batch_x, 
                                                                                            y_input: mini_batch_y, 
                                                                                            c1_input: c1, 
                                                                                            c2_input: c2, 
                                                                                            c3_input: c3, 
                                                                                            c5_input: c5}
                                                                                 )
                            
                            # Make model prediction
                            if predict_from_label == True:
                                (mini_batch_pred, _) = sess.run([prediction, last_feature_list],
                                                            feed_dict={x_input: mini_batch_x,
                                                                       c1_input: c1,
                                                                       c2_input: c2,
                                                                       c3_input: c3,
                                                                       c5_input: c5}
                                                            )
                                
                                # Get sigmoid and binary predictions
                                mini_batch_sigm = sigmoid(mini_batch_pred)
                                mini_batch_binary = np.asarray(mini_batch_sigm > threshold, dtype=np.float32)
                            
                                # Get prediction results
                                self.getResults(mini_batch_x,
                                                mini_batch_y,
                                                mini_batch_pred,
                                                mini_batch_sigm,
                                                mini_batch_binary, 
                                                mini_batch_filestr,
                                                batch_size,
                                                pos_weight,
                                                threshold,
                                                loss_type,
                                                loss_weights,
                                                myloss,
                                                0.0,
                                                mylast_feature_list, 
                                                predict_from_label,
                                                write_prediction_text_file,
                                                write_images,
                                                show_images,
                                                show_activation_maps
                                                )
                            
                            # Compute training loss
                            epoch_loss_train += myloss
                            print('Epoch=' + str(epoch) + ', iter_augs=' + str(iter_augs) + ', iter_batches=' + str(iter_batches) + ', iter_timestep=' + str(iter_timestep_apply) + ', loss=' + str(myloss) + ', epoch_loss_train=' + str(epoch_loss_train))                                

                epoch_loss_train= epoch_loss_train/(n_augs*n_evals_train/batch_size)

                
                ########## Evaluate validation loss #########
                if validation_list is not None:
                    print('\nEvaluate validation loss (epoch ' + str(epoch) + '):')
    
                    # Randomize data order
                    validation_list_copy = validation_list.copy()
                    if shuffle_sequence_order == True:
                        np.random.shuffle(validation_list_copy)
                    
                    for iter_augs in range(n_augs):
                        for iter_batches in range(n_batches_val):
                        
                            # Create composite transforms with augmentations
                            compositetransforms, horizontalflip = self.computeCompositeTransforms(validation_list_copy,
                                                                                                  batch_size_val,
                                                                                                  iter_batches,
                                                                                                  crop_size_val,
                                                                                                  k_composite,
                                                                                                  crop_from_seg_center
                                                                                                  )
                        
                            # Loop through each time step
                            for iter_timestep in range(len(validation_list_copy[iter_batches*batch_size_val]) - window_size + 1):
                                
                                # Alternate between forward and backward sequence order during validation
                                if apply_reverse_order == True:
                                    if (epoch % 2) == 0:
                                        iter_timestep_apply = iter_timestep
                                    else:
                                        iter_timestep_apply = n_timesteps_val - (iter_timestep + 1)
    
                                # Create mini batch for validation
                                mini_batch_x, mini_batch_y, mini_batch_filestr = self.createMiniBatch(validation_list_copy,
                                                                                                      batch_size_val,
                                                                                                      iter_batches,
                                                                                                      iter_timestep_apply,
                                                                                                      compositetransforms,
                                                                                                      horizontalflip,
                                                                                                      crop_size_val,
                                                                                                      window_size, 
                                                                                                      apply_augmentation,
                                                                                                      min_max
                                                                                                      )
                            
                                # Initialize convolution layers for each time step
                                if iter_timestep == 0:
                                    c1 = np.zeros(c1_input_val.get_shape())
                                    c2 = np.zeros(c2_input_val.get_shape())
                                    c3 = np.zeros(c3_input_val.get_shape())
                                    c5 = np.zeros(c5_input_val.get_shape())
                                else:
                                    c1 = mylast_feature_list[0]
                                    c2 = mylast_feature_list[1]
                                    c3 = mylast_feature_list[2]
                                    c5 = mylast_feature_list[3]
                                    
                                # Evaluate model
                                myloss, mylast_feature_list = sess.run([val_loss, last_feature_list_val],
                                                                        feed_dict={x_input_val: mini_batch_x,
                                                                                   y_input_val: mini_batch_y,
                                                                                   c1_input_val: c1,
                                                                                   c2_input_val: c2,
                                                                                   c3_input_val: c3,
                                                                                   c5_input_val: c5}
                                                                        )
                                
                                # Compute validation loss
                                epoch_loss_val += myloss
                                print('Epoch=' + str(epoch) + ', iter_augs=' + str(iter_augs) + ', iter_batches=' + str(iter_batches) + ', iter_timestep=' + str(iter_timestep_apply) + ', loss=' + str(myloss) + ', epoch_loss_val=' + str(epoch_loss_val))                                
                            
                    epoch_loss_val = epoch_loss_val/(n_augs*n_evals_val/batch_size_val)
                
                
                ########## Evaluate test loss #########
                if testing_list is not None:
                    print('\nEvaluate test loss (epoch ' + str(epoch) + '):')
                    epoch_loss_test = epoch_loss_test/n_evals_test
                    
                    
                ########## Get time elapsed, and write outputs to log #########
                end_time_epoch = datetime.datetime.now()
                log_str = 'Epoch ' + str('{:10.8f}'.format(epoch)) + ', time_elapsed: ' + str(end_time_epoch - start_time_epoch) + ' train_loss = ' + str('{:10.8f}'.format(epoch_loss_train)) + ' val_loss = ' + str('{:10.8f}'.format(epoch_loss_val)) + ' epoch_loss_test = ' + str('{:10.8f}'.format(epoch_loss_test))
                with open(self.logfilename, "a") as mylog:
                    mylog.write(log_str + '\n')
                if np.mod(epoch, epochs_save) == 0:
                    saver.save(sess, self.checkpointdirectoryname + '/' + self.checkpointprefix + "_iter" + str(epoch) + ".ckpt")
        
        return

    # %% Test model
    def test(self, testing_list,
                   model_type = 'RFCN',
                   model_checkpoint = None, 
                   restartdirectoryname = None,
                   restart_from = -1,
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
                   predict_from_label = False,
                   write_prediction_text_file = False,
                   write_images = False,
                   show_images = False,
                   show_activation_maps = False
                   ):

        # Display some settings
        print('predict_from_label:          ' + str(predict_from_label))
        print('write_prediction_text_file:  ' + str(write_prediction_text_file))
        print('write_images:                ' + str(write_images))
        print('show_images:                 ' + str(show_images))
        print('show_activation_maps:        ' + str(show_activation_maps))
        print('input_channels:              ' + str(input_channels))
        print('output_channels:             ' + str(output_channels))
        print('n_channels:                  ' + str(n_channels))
        
        # Define crop size and center
        xsize = self.training_xmax
        ysize = self.training_ymax
        crop_size = [xsize, ysize]
        
        # Define training and validation model placeholders
        with tf.name_scope('inputs'):
            x_input = tf.placeholder(tf.float32, shape=[batch_size, self.training_xmax, self.training_ymax, input_channels, window_size], name='x_input')
            y_input = tf.placeholder(tf.float32, shape=[batch_size, self.training_xmax, self.training_ymax, output_channels, window_size], name='y_input')
            c1_input = tf.placeholder(tf.float32, shape=[batch_size, xsize, ysize, n_channels], name='c1_input')
            c2_input = tf.placeholder(tf.float32, shape=[batch_size, int(np.ceil(xsize/2)), int(np.ceil(ysize/2)), n_channels*2], name='c2_input')
            c3_input = tf.placeholder(tf.float32, shape=[batch_size, int(np.ceil(xsize/4)), int(np.ceil(ysize/4)), n_channels*4], name='c3_input')
            c5_input = tf.placeholder(tf.float32, shape=[batch_size, int(np.ceil(xsize/8)), int(np.ceil(ysize/8)), n_channels*8], name='c5_input')
        
        # Build model graph
        print('>>>>>>>> Build training model graph:')
        if model_type == 'FCN':
            prediction, last_feature_list = model.fcn_outer_batchnorm(x_input, 
                                                                      c1_input, 
                                                                      c2_input, 
                                                                      c3_input, 
                                                                      c5_input, 
                                                                      input_channels=input_channels,
                                                                      output_channels=output_channels,
                                                                      n_channels=n_channels,
                                                                      training=True
                                                                      )
            
        elif model_type == 'RFCN':
            prediction, last_feature_list = model.rfcn_outer_batchnorm(x_input, 
                                                                       c1_input, 
                                                                       c2_input, 
                                                                       c3_input, 
                                                                       c5_input, 
                                                                       input_channels=input_channels,
                                                                       output_channels=output_channels,
                                                                       n_channels=n_channels,
                                                                       training=True
                                                                       )
           
        
        # Write out the model layer sizes
        prediction = tf.stack(prediction, axis=4) # tf.concat(prediction, axis=4)
        print('prediction (after stack): ' + str(prediction.shape))
        
        # Define loss for testing (different due to image patch size, hardcoded)
        print('Loss function: ' + str(loss_type))
        test_loss = self.setLoss_TF(logits=prediction, labels=y_input, loss_type=loss_type, loss_weights=loss_weights, pos_weight=pos_weight)   
            
        # Get number of sequences, timesteps per sequence, batch iterations per epoch, and total number of evaluations
        n_sequences = len(testing_list)
        n_batches = int(n_sequences/batch_size)
        
        # Build variables lists
        vnet_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "contracting_path") + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "expanding_path")
        vnet_saver = tf.train.Saver(vnet_vars) # Model loader
        saver = tf.train.Saver() # RFCN saver
        
        ########################################### Start test session ###########################################  
        config = tf.ConfigProto(device_count={'GPU':1})
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            if model_checkpoint is not None:
                vnet_saver.restore(sess, model_checkpoint)
            
            if restart_from > -1:
                if restartdirectoryname is None:
                    saver.restore(sess, str(self.checkpointdirectoryname) + '/' + str(self.checkpointprefix) + "_iter" + str(restart_from) + ".ckpt")
                else:
                    saver.restore(sess, restartdirectoryname)

            ######### Evaluate model on test data #########
            for iter_batches in range(n_batches):
                print('\n>>>> Test next batch >>>>')
                
                # Compute composite transforms with data augmentation
                compositetransforms, horizontalflip = self.computeCompositeTransforms(testing_list,
                                                                                      batch_size,
                                                                                      iter_batches,
                                                                                      crop_size,
                                                                                      k_composite,
                                                                                      crop_from_seg_center
                                                                                      )
                
                # Loop through each time step
                for iter_timestep in range(len(testing_list[iter_batches*batch_size]) - window_size + 1):
                    
                    start_time_batch = datetime.datetime.now()
                    
                    # Create mini batch
                    mini_batch_x, mini_batch_y, mini_batch_filestr = self.createMiniBatch(testing_list,
                                                                                          batch_size,
                                                                                          iter_batches,
                                                                                          iter_timestep,
                                                                                          compositetransforms,
                                                                                          horizontalflip,
                                                                                          crop_size,
                                                                                          window_size, 
                                                                                          apply_augmentation,
                                                                                          min_max
                                                                                          )

                    # Initialize convolution layers for each time step
                    if iter_timestep == 0:
                        c1 = np.zeros(c1_input.get_shape())
                        c2 = np.zeros(c2_input.get_shape())
                        c3 = np.zeros(c3_input.get_shape())
                        c5 = np.zeros(c5_input.get_shape())
                    else:
                        c1 = mylast_feature_list[0]
                        c2 = mylast_feature_list[1]
                        c3 = mylast_feature_list[2]
                        c5 = mylast_feature_list[3]
                        
                    # Make model prediction
                    start_time_batch_predict = datetime.datetime.now()
                    mini_batch_pred, mylast_feature_list = sess.run([prediction, last_feature_list],
                                                                    feed_dict={x_input: mini_batch_x,
                                                                               c1_input: c1,
                                                                               c2_input: c2,
                                                                               c3_input: c3,
                                                                               c5_input: c5}
                                                                    )
                    end_time_batch_predict = datetime.datetime.now()
                    time_elapsed_batch_predict = end_time_batch_predict - start_time_batch_predict
                    print('Time (predict batch), ms=' + str(time_elapsed_batch_predict.total_seconds() * 1000))
                    
                    # Get sigmoid and binary predictions
                    mini_batch_sigm = sigmoid(mini_batch_pred)
                    mini_batch_binary = np.asarray(mini_batch_sigm > threshold, dtype=np.float32)
                    
                    # Evaluate model
                    if predict_from_label == True:  
                        (myloss, _) = sess.run([test_loss, last_feature_list],
                                           feed_dict={x_input: mini_batch_x,
                                                      y_input: mini_batch_y,
                                                      c1_input: c1,
                                                      c2_input: c2,
                                                      c3_input: c3,
                                                      c5_input: c5}
                                         )
                    else:
                        myloss = 0.0
                    
                    print('myloss:')
                    print(myloss)
                    self.getResults(mini_batch_x,
                                    mini_batch_y,
                                    mini_batch_pred,
                                    mini_batch_sigm,
                                    mini_batch_binary,
                                    mini_batch_filestr,
                                    batch_size,
                                    pos_weight,
                                    threshold,
                                    loss_type,
                                    loss_weights,
                                    myloss,
                                    time_elapsed_batch_predict,
                                    mylast_feature_list,
                                    predict_from_label,
                                    write_prediction_text_file,
                                    write_images,
                                    show_images,
                                    show_activation_maps
                                    )

                    end_time_batch = datetime.datetime.now()
                    delta_time_batch = end_time_batch - start_time_batch
                    print('Time (iter batch), ms=' + str(delta_time_batch.total_seconds() * 1000))
            
        return
    
    