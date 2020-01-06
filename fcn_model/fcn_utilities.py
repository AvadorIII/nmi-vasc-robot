# -*- coding: utf-8 -*-
"""
Authors: A. Chen, M. Balter, T. Maguire, M. Yarmush
Affiliation: Rutgers, The State University of New Jersey
Last updated: January 6, 2020

"""

import re
import numpy as np
import datetime
import imageio
import matplotlib.pyplot as plt


def countList2D(list):
    nitems = 0
    for i in range(len(list)):
        nitems += len(list[i])
    
    return nitems


def multireplace(string, replacements):
    """
    Given a string and a replacement map, it returns the replaced string.
    :param str string: string to execute replacements on
    :param dict replacements: replacement dictionary {value to find: value to replace}
    :rtype: str
    """
    # Place longer ones first to keep shorter substrings from matching where the longer ones should take place
    # For instance given the replacements {'ab': 'AB', 'abc': 'ABC'} against the string 'hey abc', it should produce
    # 'hey ABC' and not 'hey ABc'
    substrs = sorted(replacements, key=len, reverse=True)

    # Create a big OR regex that matches any of the substrings to replace
    regexp = re.compile('|'.join(map(re.escape, substrs)))

    # For each match, look up the new string in the replacements
    return regexp.sub(lambda match: replacements[match.group(0)], string)

    
def showActivationMaps(mylast_feature_list, _figsize, _grid_yn):

    # c1 layers
    n_img = mylast_feature_list[0].shape[3]
    print('c1, n_img=' + str(n_img))
    fig, axs = plt.subplots(1, n_img, figsize=(n_img*_figsize, _figsize))
    for i in range(n_img):
        axs[i].imshow(mylast_feature_list[0][0,:,:,i], cmap='gray')
        axs[i].grid(_grid_yn)
    plt.show()
    
    #c2 layers
    print('c2, n_img=' + str(n_img))
    n_img = mylast_feature_list[1].shape[3]
    fig, axs = plt.subplots(1, n_img, figsize=(0.5*n_img*_figsize, _figsize/2))
    for i in range(n_img):
        axs[i].imshow(mylast_feature_list[1][0,:,:,i], cmap='gray')
        axs[i].grid(_grid_yn)
    plt.show()
    
    #c3 layers
    print('c3, n_img=' + str(n_img))
    n_img = mylast_feature_list[2].shape[3]
    fig, axs = plt.subplots(1, n_img, figsize=(0.5*n_img*_figsize, _figsize/2))
    for i in range(n_img):
        axs[i].imshow(mylast_feature_list[2][0,:,:,i], cmap='gray')
        axs[i].grid(_grid_yn)
    plt.show()
    
    #c5 layers
    print('c5, n_img=' + str(n_img))
    n_img = mylast_feature_list[3].shape[3]
    fig, axs = plt.subplots(1, n_img, figsize=(0.5*n_img*_figsize, _figsize/2))
    for i in range(n_img):
        axs[i].imshow(mylast_feature_list[3][0,:,:,i], cmap='gray')
        axs[i].grid(_grid_yn)
    plt.show()

    return


def showImagesAsSubplots(image_list, title_list, _figsize, n_rows=1, min_max=(0,1), _grid_yn=True, filestr=None):
    start_time = datetime.datetime.now()
    
    n_img = round(len(image_list) / n_rows)
    fig, axs = plt.subplots(n_rows, n_img, figsize=_figsize)
    if n_rows == 1:
        for i in range(n_img):
            axs[i].imshow(image_list[i])
            axs[i].set_title(title_list[i])
            axs[i].grid(_grid_yn)        
    elif n_rows >= 2:
        for j in range(n_rows):
            for i in range(n_img):
                idx = n_img * j + i
                axs[j, i].imshow(image_list[idx], vmin=min_max[0], vmax=min_max[1])
                axs[j, i].set_title(title_list[idx])
                axs[j, i].grid(_grid_yn)
            
    for ax in axs.flat:
        ax.label_outer()
    plt.tight_layout()    
    plt.show()
    plt.draw()
    if filestr is not None:
        fig.savefig(filestr, dpi=200)
        
    end_time = datetime.datetime.now()
    time_elapsed = end_time - start_time
    print('Time (show subplots), ms=' + str(time_elapsed.total_seconds() * 1000))
        
        
def writeOutputsToFile(out_filestr, img_name, iter_plot, mini_batch_x, mini_batch_y, mini_batch_pred, mini_batch_sigm, mini_batch_binary, write_raw_data=False):
    start_time = datetime.datetime.now()
    print(out_filestr)
    
    for i in range(mini_batch_x.shape[3]):
        data = mini_batch_x[iter_plot,:,:,i,0]
        imageio.imwrite(out_filestr + '\\' + img_name + '_image_'  + str(i) + '.png', data)
        #print('Image (min, max)=' + str(np.amin(data)) + ', ' + str(np.amax(data)))
    
    for i in range(mini_batch_y.shape[3]):    
        data = mini_batch_y[iter_plot,:,:,i,0]
        imageio.imwrite(out_filestr + '\\' + img_name + '_label_'  + str(i) + '.png', data)
        #print('Label (min, max)=' + str(np.amin(data)) + ', ' + str(np.amax(data)))
    
    for i in range(mini_batch_pred.shape[3]):    
        data = mini_batch_pred[iter_plot,:,:,i,0]
        imageio.imwrite(out_filestr + '\\' + img_name + '_pred_'  + str(i) + '.png', data)
        #print('Pred (min, max)=' + str(np.amin(data)) + ', ' + str(np.amax(data)))
    
    for i in range(mini_batch_sigm.shape[3]):         
        data = mini_batch_sigm[iter_plot,:,:,i,0]
        imageio.imwrite(out_filestr + '\\' + img_name + '_sigm_'  + str(i) + '.png', data)
        #print('Sigmoid (min, max)=' + str(np.amin(data)) + ', ' + str(np.amax(data)))
    
    for i in range(mini_batch_binary.shape[3]):         
        data = mini_batch_binary[iter_plot,:,:,i,0]
        imageio.imwrite(out_filestr + '\\' + img_name + '_binary_'  + str(i) + '.png', data)
        #print('Binary (min, max)=' + str(np.amin(data)) + ', ' + str(np.amax(data)))
        
    if write_raw_data == True:
        np.save(out_filestr + '\\' + img_name + '_image',  mini_batch_x)
        np.save(out_filestr + '\\' + img_name + '_label',  mini_batch_y)
        np.save(out_filestr + '\\' + img_name + '_pred',   mini_batch_pred)
        np.save(out_filestr + '\\' + img_name + '_sigm',   mini_batch_sigm)
        np.save(out_filestr + '\\' + img_name + '_binary', mini_batch_binary)
        
    end_time = datetime.datetime.now()
    time_elapsed = end_time - start_time
    print('Time (write images), ms=' + str(time_elapsed.total_seconds() * 1000))