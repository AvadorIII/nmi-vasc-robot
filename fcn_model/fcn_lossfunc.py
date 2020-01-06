# -*- coding: utf-8 -*-
"""
Authors: A. Chen, M. Balter, T. Maguire, M. Yarmush
Affiliation: Rutgers, The State University of New Jersey
Last updated: January 6, 2020

"""

import tensorflow as tf
import numpy as np


def sigmoid(x, derivative=False):
    sigm = 1. / (1. + np.exp(-x))
    if derivative:
        return sigm * (1. - sigm)
    return sigm


def computeModifiedHausdorffDistance2D_TF(points1, points2): # find nearest neighbors from points1 to points2
    pointdistmat = tf.sqrt((tf.tile(tf.reshape(points1[:, 0], (points1.get_shape()[0].value, 1)), (1, points2.get_shape()[0].value)) - tf.tile(tf.reshape(points2[:, 0], (1, points2.get_shape()[0].value)), (points1.get_shape()[0].value, 1)))**2 + (tf.tile(tf.reshape(points1[:, 1], (points1.get_shape()[0].value, 1)), (1, points2.get_shape()[0].value)) - tf.tile(tf.reshape(points2[:, 1], (1, points2.get_shape()[0].value)), (points1.get_shape()[0].value, 1)))**2 )
    dist12 = tf.mean(tf.minimum(pointdistmat, axis=0))
    dist21 = tf.mean(tf.minimum(pointdistmat, axis=1))
    return np.maximum([dist12, dist21])


def computeModifiedHausdorffDistance2D(points1, points2): # find nearest neighbors from points1 to points2
    pointdistmat = np.sqrt((np.tile(np.reshape(points1[:, 0], (points1.shape[0], 1)), (1, points2.shape[0])) - np.tile(np.reshape(points2[:, 0], (1, points2.shape[0])), (points1.shape[0], 1)))**2 + (np.tile(np.reshape(points1[:, 1], (points1.shape[0], 1)), (1, points2.shape[0])) - np.tile(np.reshape(points2[:, 1], (1, points2.shape[0])), (points1.shape[0], 1)))**2 )
    dist12 = np.mean(np.min(pointdistmat, axis=0))
    dist21 = np.mean(np.min(pointdistmat, axis=1))
    return np.max([dist12, dist21])


def computeWeightedCrossEntropyWithLogits_TF(logits, labels, pos_weight=1.0):
    sgmd = tf.nn.sigmoid(logits)
    loss = -tf.reduce_mean(labels * tf.log(sgmd) * pos_weight) - tf.reduce_mean((1 - labels) * tf.log(1 - sgmd))
    return loss


def computeWeightedCrossEntropyWithLogits(logits, labels, pos_weight=1.0):
    sgmd = sigmoid(logits)
    loss = -np.sum(labels * np.log(sgmd) * pos_weight) - np.sum((1 - labels) * np.log(1 - sgmd))
    return loss


def computeDiceLoss_TF(logits, labels):
    smooth = 1e-7
    logits = tf.math.sigmoid(logits)
    numerator = tf.reduce_sum(labels * logits)
    denominator = tf.reduce_sum(labels + logits)
    dice = 2.0*(numerator + smooth) / (denominator + smooth)
    loss = 1.0 - dice
    return loss


def computeGeneralizedDiceLoss2D_TF(logits, labels):
    smooth = 1e-7
    shape = tf.TensorShape(logits.shape).as_list()
    depth = int(shape[-1])
    labels = tf.cast(labels, tf.int32)
    labels = tf.one_hot(labels, depth, dtype=tf.float32)
    logits = tf.nn.softmax(logits)
    weights = 1.0 / (tf.reduce_sum(labels, axis=[0, 1, 2])**2 + smooth)
    numerator = tf.reduce_sum(labels * logits, axis=[0, 1, 2])
    numerator = tf.reduce_sum(weights * numerator)
    denominator = tf.reduce_sum(labels + logits, axis=[0, 1, 2])
    denominator = tf.reduce_sum(weights * denominator)
    dice = 2.0 * (numerator + smooth) / (denominator + smooth)
    loss = 1.0 - dice
    return loss


def computeGeneralizedDiceLoss_TF(logits, labels, axis_list):
    smooth = 1e-7
    logits = tf.nn.softmax(logits)
    weights = 1.0 / (tf.reduce_sum(labels, axis=axis_list)**2 + smooth)
    numerator = tf.reduce_sum(labels * logits, axis=axis_list)
    numerator = tf.reduce_sum(weights * numerator)
    denominator = tf.reduce_sum(labels + logits, axis=axis_list)
    denominator = tf.reduce_sum(weights * denominator)
    dice = 2.0*(numerator + smooth) / (denominator + smooth)
    loss = 1.0 - dice
    return loss


def computeDiceScorePrediction(predict, target):
    smooth = 1e-7
    predict[predict > 1] = 1.0
    target[target > 1] = 1.0
    intersection = predict.copy()
    intersection[target == 0] = 0.0
    dice = 2.0 * (np.sum(intersection) + smooth) / (np.sum(target) + np.sum(predict) + smooth)
    return dice


def computeMeanSquaredLoss_TF(logits, labels, weights=1.0):
    loss = weights * tf.reduce_mean(tf.square(logits - labels))
    return loss


def computeMeanSquaredLoss(logits, labels, weights=1.0):
    loss = weights * np.mean(np.square(logits - labels))
    return loss


def evalLoss_customized(logits, labels, pos_weight, threshold, calc_TF=False, calc_regularizers=False):
    
    calcWeightedCrossEntropy   = 0.0
    calcLossDiceSoft           = 0.0
    calcLossDiceHard           = 0.0
    calcMeanSquares            = 0.0
    calcImageTotalVariation_TF = 0.0
    calcImageGradientL2_TF     = 0.0
    
    # Display raw value parameters
    print('logits (shape)=' + str(logits.shape) + ', labels (shape)=' + str(logits.shape))
    print('logits (min, max)=' + str(np.amin(logits)) + ', ' + str(np.amax(logits)) + '; labels (min, max)=' + str(np.amin(labels)) + ', ' + str(np.amax(labels)))
    print('logits (reduce_mean)=' + str(np.mean(logits)) + '; labels (reduce_mean)=' + str(np.mean(labels)))
    
    # For classification loss
    logits_sigmoid = sigmoid(logits)
    logits_binary = (logits_sigmoid > threshold).astype(np.float32)
    labels_binary = (labels > threshold).astype(np.float32)
    
    if calc_TF == True:
        
        # Cross entropy loss
        calcCrossEntropy_TF = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels_binary)).eval()
        calcWeightedCrossEntropy_TF = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=labels_binary, logits=logits, pos_weight=pos_weight)).eval()
        calcWeightedCrossEntropy = computeWeightedCrossEntropyWithLogits(logits=logits, labels=labels_binary, pos_weight=pos_weight) # need to figure out why this outputs NaN
        print('calcCrossEntropy_TF=' + str(calcCrossEntropy_TF))
        print('calcWeightedCrossEntropy_TF=' + str(calcWeightedCrossEntropy_TF))
        print('calcWeightedCrossEntropy=' + str(calcWeightedCrossEntropy))
        
        # Dice loss
        calcLossDiceSoft_TF = computeDiceLoss_TF(logits, labels_binary).eval()
        calcLossDiceSoft = computeDiceScorePrediction(logits_sigmoid, labels_binary)
        calcLossDiceHard = computeDiceScorePrediction(logits_binary, labels_binary)
        print('calcLossDiceSoft_TF, loss=' + str(calcLossDiceSoft_TF) + ', score=' + str(1 - calcLossDiceSoft_TF))
        print('calcLossDiceSoft=' + str(calcLossDiceSoft))
        print('calcLossDiceHard=' + str(calcLossDiceHard))
        
        # Mean squared loss
        calcMeanSquares_TF = tf.losses.mean_squared_error(labels=labels, predictions=logits, weights=1.0).eval()
        calcMeanSquares_TF2 = computeMeanSquaredLoss_TF(logits, labels, weights=1.0).eval()
        calcMeanSquares = computeMeanSquaredLoss(logits, labels, weights=1.0)
        print('calcMeanSquares_TF=' + str(calcMeanSquares_TF))
        print('calcMeanSquares_TF2=' + str(calcMeanSquares_TF2))
        print('calcMeanSquares=' + str(calcMeanSquares))
        
    else:
        # Cross entropy loss
        calcWeightedCrossEntropy = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=labels_binary, logits=logits, pos_weight=pos_weight)).eval() # still using TF function here
        print('calcWeightedCrossEntropy=' + str(calcWeightedCrossEntropy))
        
        # Dice loss
        calcLossDiceSoft = computeDiceScorePrediction(logits_sigmoid, labels_binary)
        calcLossDiceHard = computeDiceScorePrediction(logits_binary, labels_binary)
        print('calcLossDiceSoft=' + str(calcLossDiceSoft))
        print('calcLossDiceHard=' + str(calcLossDiceHard))
        
        # Mean squared loss
        calcMeanSquares = computeMeanSquaredLoss(logits, labels, weights=1.0)
        print('calcMeanSquares=' + str(calcMeanSquares))
        
    if calc_regularizers == True:
        
        logits_expand = logits
        if logits_expand.ndim == 2:
            logits_expand = np.expand_dims(logits_expand, 0)
            logits_expand = np.expand_dims(logits_expand, logits_expand.ndim)
        elif logits_expand.ndim == 3:
            logits_expand = np.expand_dims(logits_expand, logits_expand.ndim)
        
        # L1 loss (total variation of neighboring pixels)
        calcImageTotalVariation_TF = tf.reduce_sum(tf.image.total_variation(tf.convert_to_tensor(logits_expand))) / tf.cast(tf.size(logits), dtype=tf.float32)
        calcImageTotalVariation_TF = calcImageTotalVariation_TF.eval()
        print('calcImageTotalVariation_TF=' + str(calcImageTotalVariation_TF))
        
        # L2 image loss (Squared magnitude difference of neighboring pixels)        
        (grady, gradx) = tf.image.image_gradients(tf.convert_to_tensor(logits_expand))
        calcImageGradientL2_TF = tf.reduce_mean(tf.square(grady)) + tf.reduce_mean(tf.square(gradx))
        calcImageGradientL2_TF = calcImageGradientL2_TF.eval()
        print('calcImageGradientL2_TF=' + str(calcImageGradientL2_TF))
    
    return (calcWeightedCrossEntropy, calcLossDiceSoft, calcLossDiceHard, calcMeanSquares, calcImageTotalVariation_TF, calcImageGradientL2_TF)