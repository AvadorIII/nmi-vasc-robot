# -*- coding: utf-8 -*-
"""
Authors: A. Chen, M. Balter, T. Maguire, M. Yarmush
Affiliation: Rutgers, The State University of New Jersey
Last updated: January 6, 2020

"""

#%% Import packages

import os

codepath = os.getcwd()
datapath = '..\\data\\dus_test\\'

datalist_test = [[datapath + '\\sequence1\\', -1, 0, 9999, 4],
                 [datapath + '\\sequence2\\', -1, 0, 9999, 4],
                 [datapath + '\\sequence3\\', -1, 0, 9999, 4],
                 [datapath + '\\sequence4\\', -1, 0, 9999, 4],
                 [datapath + '\\sequence5\\', -1, 0, 9999, 4],
                 [datapath + '\\sequence6\\', -1, 0, 9999, 4],
                 [datapath + '\\sequence7\\', -1, 0, 9999, 4],
                 [datapath + '\\sequence8\\', -1, 0, 9999, 4],
                 [datapath + '\\sequence9\\', -1, 0, 9999, 4],
                 [datapath + '\\sequence10\\', -1, 0, 9999, 4],
                 [datapath + '\\sequence11\\', -1, 0, 9999, 4]]

datalist_val = datalist_test # for code demonstration purposes, use the test dataset
datalist_train = datalist_test # for code demonstration purposes, use the test dataset

fileprefix = 'data_'
filesuffix = '.mat'

# %% Data parameters

context_length_train = 10
context_length_val = 10
context_length_test = 10

context_backtrack_train = 0.6
context_backtrack_val = 0.6
context_backtrack_test = 0.0

# %% Generate training file list (for RFCN)
listfilename = codepath + '\\' + 'datalists' + '\\' + 'training_list_rfcn.txt'

with open(listfilename,'w') as file:
    for ii in range(len(datalist_train)):
        
        directoryname = datalist_train[ii][0]
        curind = datalist_train[ii][2] + 1
        maxind = datalist_train[ii][3]
        timesteps = datalist_train[ii][4]
        
        curind_start = curind
        while True: # if negative number, starts from frame zero
            curind_inner = 0
            
            while (curind_inner < context_length_train) & (curind <= maxind): # fill up the current time block
                
                curind_timestep = 0
                while (curind_timestep < timesteps) & (curind <= maxind): # allows option to skip frames
                    file_name = directoryname + fileprefix + '{:06d}'.format(curind) + filesuffix
                    if os.path.isfile(file_name):
                        file_name_last_good = file_name
                        curind_timestep += 1
                    curind += 1
                    
                print('>>>>>>>> ' + file_name_last_good)
                file.write(file_name_last_good + '\n')
                curind_inner += 1
                
            # If the sequence ends before fully reaching the context length, add previous frames to the block until filled up
            # Do this only if most of the block is filled up
            if (curind_inner < context_length_train) & (curind_inner >= int(0.8 * context_length_train)):
                curind_backward = curind - 2
                while (curind_inner < context_length_train):
                    file_name = directoryname + fileprefix + '{:06d}'.format(curind_backward) + filesuffix
                    if os.path.isfile(file_name):
                        print(file_name)
                        file.write(file_name + '\n')    
                        curind_inner += 1
                    curind_backward -= 1
            
            file.write('-\n')
            
            # Set how many frames to backtrack for start of next time block
            # Set to zero if we don't want any backtracking (i.e. no frames overlap from one time block to the next)
            if curind > maxind:
                break
            else:
                print('curind (before): ' + str(curind))
                curind_backtrack = 0
                num_frames_to_backtrack = timesteps * context_length_train * context_backtrack_train
                while (curind_backtrack < num_frames_to_backtrack) & (curind > curind_start):
                    file_name = directoryname + fileprefix + '{:06d}'.format(curind) + filesuffix
                    if os.path.isfile(file_name):
                        curind_backtrack += 1
                    curind -= 1
                print('curind (after): ' + str(curind))
            
    file.close()
    
# %% Generate validation file list (for RFCN)

listfilename = codepath + '\\' + 'datalists' + '\\' + 'validation_list_rfcn.txt'

with open(listfilename,'w') as file:
    for ii in range(len(datalist_val)):
        
        directoryname = datalist_val[ii][0]
        curind = datalist_val[ii][1]
        maxind = datalist_val[ii][2]
        timesteps = datalist_val[ii][4]
        
        curind_start = curind
        while True: # if negative number, starts from frame zero
            curind_inner = 0
            
            while (curind_inner < context_length_val) & (curind <= maxind): # fill up the current time block
                
                curind_timestep = 0
                while (curind_timestep < timesteps) & (curind <= maxind): # allows option to skip frames
                    file_name = directoryname + fileprefix + '{:06d}'.format(curind) + filesuffix
                    if os.path.isfile(file_name):
                        file_name_last_good = file_name
                        curind_timestep += 1
                    curind += 1
                    
                print('>>>>>>>> ' + file_name_last_good)
                file.write(file_name_last_good + '\n')
                curind_inner += 1
                
            # If the sequence ends before fully reaching the context length, add previous frames to the block until filled up
            # Do this only if most of the block is filled up
            if (curind_inner < context_length_val) & (curind_inner >= int(0.8 * context_length_val)):
                curind_backward = curind - 2
                while (curind_inner < context_length_val):
                    file_name = directoryname + fileprefix + '{:06d}'.format(curind_backward) + filesuffix
                    if os.path.isfile(file_name):
                        print(file_name)
                        file.write(file_name + '\n')    
                        curind_inner += 1
                    curind_backward -= 1
            
            file.write('-\n')
            
            # Set how many frames to backtrack for start of next time block
            # Set to zero if we don't want any backtracking (i.e. no frames overlap from one time block to the next)
            if curind > maxind:
                break
            else:
                print('curind (before): ' + str(curind))
                curind_backtrack = 0
                num_frames_to_backtrack = timesteps * context_length_val * context_backtrack_val
                while (curind_backtrack < num_frames_to_backtrack) & (curind > curind_start):
                    file_name = directoryname + fileprefix + '{:06d}'.format(curind) + filesuffix
                    if os.path.isfile(file_name):
                        curind_backtrack += 1
                    curind -= 1
                print('curind (after): ' + str(curind))
            
    file.close()

# %% Generate testing file list (for RFCN)

listfilename = codepath + '\\' + 'datalists' + '\\' + 'testing_list_rfcn.txt'

with open(listfilename,'w') as file:
    for ii in range(len(datalist_test)):
        
        directoryname = datalist_test[ii][0]
        curind = datalist_test[ii][2] + 1
        maxind = datalist_test[ii][3]
        timesteps = datalist_test[ii][4]
        
        curind_start = curind
        while True: # if negative number, starts from frame zero
            curind_inner = 0
            
            while (curind_inner < context_length_test) & (curind <= maxind): # fill up the current time block
                
                curind_timestep = 0
                while (curind_timestep < timesteps) & (curind <= maxind): # allows option to skip frames
                    file_name = directoryname + fileprefix + '{:06d}'.format(curind) + filesuffix
                    if os.path.isfile(file_name):
                        file_name_last_good = file_name
                        curind_timestep += 1
                    curind += 1
                    
                print('>>>>>>>> ' + file_name_last_good)
                file.write(file_name_last_good + '\n')
                curind_inner += 1
                
            # If the sequence ends before fully reaching the context length, add previous frames to the block until filled up
            # Do this only if most of the block is filled up
            if (curind_inner < context_length_test) & (curind_inner >= int(0.8 * context_length_test)):
                curind_backward = curind - 2
                while (curind_inner < context_length_test):
                    file_name = directoryname + fileprefix + '{:06d}'.format(curind_backward) + filesuffix
                    if os.path.isfile(file_name):
                        print(file_name)
                        file.write(file_name + '\n')    
                        curind_inner += 1
                    curind_backward -= 1
            
            file.write('-\n')
            
            # Set how many frames to backtrack for start of next time block
            # Set to zero if we don't want any backtracking (i.e. no frames overlap from one time block to the next)
            if curind > maxind:
                break
            else:
                print('curind (before): ' + str(curind))
                curind_backtrack = 0
                num_frames_to_backtrack = timesteps * context_length_test * context_backtrack_test
                while (curind_backtrack < num_frames_to_backtrack) & (curind > curind_start):
                    file_name = directoryname + fileprefix + '{:06d}'.format(curind) + filesuffix
                    if os.path.isfile(file_name):
                        curind_backtrack += 1
                    curind -= 1
                print('curind (after): ' + str(curind))
            
    file.close()