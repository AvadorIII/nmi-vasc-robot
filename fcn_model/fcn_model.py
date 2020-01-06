# -*- coding: utf-8 -*-
"""
Authors: A. Chen, M. Balter, T. Maguire, M. Yarmush
Affiliation: Rutgers, The State University of New Jersey
Last updated: January 6, 2020

"""

import tensorflow as tf


def prelu(x): # Parametric leaky Relu
    alpha = tf.get_variable('alpha', shape=x.get_shape()[-1], dtype=x.dtype, initializer=tf.constant_initializer(0.1))
    return tf.maximum(0.0, x) + alpha * tf.minimum(0.0, x)


def convolution_2d(layer_input, filter, strides, padding='SAME'):
    assert len(filter) == 4  # [filter_height, filter_width, in_channels, out_channels]
    assert len(strides) == 4  # must match input dimensions [batch, in_height, in_width, in_channels]
    assert padding in ['VALID', 'SAME']
    w = tf.get_variable('weights',shape=filter,initializer=tf.contrib.layers.xavier_initializer(),trainable=True)
    b = tf.get_variable('biases',shape=[filter[-1]],initializer=tf.ones_initializer)
    return tf.nn.conv2d(layer_input, w, strides, padding) + b


def deconvolution_2d(layer_input, filter, output_shape, strides, padding='SAME'):
    assert len(filter) == 4  # [depth, height, width, output_channels, in_channels]
    assert len(strides) == 4  # must match input dimensions [batch, depth, height, width, in_channels]
    assert padding in ['VALID', 'SAME']
    w = tf.get_variable('weights',shape=filter,initializer=tf.contrib.layers.xavier_initializer(),trainable=True)
    b = tf.get_variable('biases',shape=[filter[-2]],initializer=tf.ones_initializer)
    return tf.nn.conv2d_transpose(layer_input, w, output_shape, strides, padding) + b


def convGRU_2d_gate(layer_input, previous_state, filter, strides, padding='SAME'):
    assert len(filter) == 4  # [filter_depth, filter_height, filter_width, in_channels, out_channels]
    assert len(strides) == 4  # must match input dimensions [batch, in_depth, in_height, in_width, in_channels]
    assert padding in ['VALID', 'SAME']
    w = tf.get_variable('w_weights',shape=filter,initializer=tf.contrib.layers.xavier_initializer(),trainable=True)
    u = tf.get_variable('u_weights',shape=filter,initializer=tf.contrib.layers.xavier_initializer(),trainable=True)
    b = tf.get_variable('biases',shape=[filter[-1]],initializer=tf.zeros_initializer(),trainable=True)
    return tf.nn.sigmoid(tf.nn.conv2d(layer_input, w, strides, padding) + tf.nn.conv2d(previous_state, u, strides, padding) + b)


def convGRU_2d_output(layer_input, previous_state, z_gate, r_gate, filter, strides, padding='SAME'):
    assert len(filter) == 4  # [filter_depth, filter_height, filter_width, in_channels, out_channels]
    assert len(strides) == 4  # must match input dimensions [batch, in_depth, in_height, in_width, in_channels]
    assert padding in ['VALID', 'SAME']
    w = tf.get_variable('w_weights',shape=filter,initializer=tf.contrib.layers.xavier_initializer(),trainable=True)
    u = tf.get_variable('u_weights',shape=filter,initializer=tf.contrib.layers.xavier_initializer(),trainable=True)
    b = tf.get_variable('biases',shape=[filter[-1]],initializer=tf.zeros_initializer(),trainable=True)
    return (1-z_gate) * previous_state + z_gate * tf.nn.tanh(tf.nn.conv2d(layer_input, w, strides, padding) + tf.nn.conv2d(previous_state*r_gate, u, strides, padding) + b)


def convolution_block(layer_input, n_channels, num_convolutions, reuse=False):
    x = layer_input
    for i in range(num_convolutions - 1):
        with tf.variable_scope('conv_' + str(i+1), reuse=reuse):
            x = convolution_2d(x, [5, 5, n_channels, n_channels], [1, 1, 1, 1])
            x = prelu(x)
    x = convolution_2d(x, [5, 5, n_channels, n_channels], [1, 1, 1, 1])
    x = x + layer_input
    return prelu(x)


def convolution_block_2(layer_input, fine_grained_features, n_channels, num_convolutions,reuse=False):
    x = tf.concat((layer_input, fine_grained_features), axis=-1)
    with tf.variable_scope('conv_' + str(1), reuse=reuse):
        x = convolution_2d(x, [5, 5, n_channels * 2, n_channels], [1, 1, 1, 1])

    for i in range(1, num_convolutions - 1):
        with tf.variable_scope('conv_' + str(i+1), reuse=reuse):
            x = convolution_2d(x, [5, 5, n_channels, n_channels], [1, 1, 1, 1])
            x = prelu(x)

    x = convolution_2d(x, [5, 5, n_channels, n_channels], [1, 1, 1, 1])
    x = x + layer_input
    return prelu(x)


def convolution_block_in(layer_input, n_channels, num_convolutions, kx, ky, reuse=False):
    x = layer_input
    for i in range(num_convolutions - 1):
        with tf.variable_scope('conv_' + str(i+1), reuse=reuse):
            x = convolution_2d(x, [kx, ky, n_channels, n_channels], [1, 1, 1, 1])
            x = prelu(x)
    x = convolution_2d(x, [kx, ky, n_channels, n_channels], [1, 1, 1, 1])
    x = x + layer_input
    return prelu(x)


def convolution_block_in_batchnorm(layer_input, n_channels, num_convolutions, kx, ky, reuse=False, training=True):
    x = layer_input
    for i in range(num_convolutions - 1):
        with tf.variable_scope('conv_' + str(i+1), reuse=reuse):
            x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001, center=True, scale=True, training=training)
            x = convolution_2d(x, [kx, ky, n_channels, n_channels], [1, 1, 1, 1])
            x = prelu(x)
            
    x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001, center=True, scale=True, training=training)
    x = convolution_2d(x, [kx, ky, n_channels, n_channels], [1, 1, 1, 1])
    x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001, center=True, scale=True, training=training)
    x = x + layer_input
    return prelu(x)


def convolution_block_2_in(layer_input, fine_grained_features, n_channels, num_convolutions, kx, ky, reuse=False):
    x = tf.concat((layer_input, fine_grained_features), axis=-1)
    with tf.variable_scope('conv_' + str(1), reuse=reuse):
        x = convolution_2d(x, [kx, ky, n_channels * 2, n_channels], [1, 1, 1, 1])

    for i in range(1, num_convolutions - 1):
        with tf.variable_scope('conv_' + str(i+1), reuse=reuse):
            x = convolution_2d(x, [kx, ky, n_channels, n_channels], [1, 1, 1, 1])
            x = prelu(x)

    x = convolution_2d(x, [kx, ky, n_channels, n_channels], [1, 1, 1, 1])
    x = x + layer_input
    return prelu(x)


def convolution_block_2_in_batchnorm(layer_input, fine_grained_features, n_channels, num_convolutions, kx, ky, reuse=False, training=True):
    x = tf.concat((layer_input, fine_grained_features), axis=-1)
    with tf.variable_scope('conv_' + str(1), reuse=reuse):
        x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001, center=True, scale=True, training=training)
        x = convolution_2d(x, [kx, ky, n_channels * 2, n_channels], [1, 1, 1, 1])
        
    for i in range(1, num_convolutions - 1):
        with tf.variable_scope('conv_' + str(i+1), reuse=reuse):
            x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001, center=True, scale=True, training=training)
            x = convolution_2d(x, [kx, ky, n_channels, n_channels], [1, 1, 1, 1])
            x = prelu(x)

    x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001, center=True, scale=True, training=training)
    x = convolution_2d(x, [kx, ky, n_channels, n_channels], [1, 1, 1, 1])
    x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001, center=True, scale=True, training=training)
    x = x + layer_input
    return prelu(x)


def down_convolution(layer_input, in_channels, reuse=False):
    with tf.variable_scope('down_convolution', reuse=reuse):
        x = convolution_2d(layer_input, [2, 2, in_channels, in_channels * 2], [1, 2, 2, 1])
        return prelu(x)


def up_convolution(layer_input, output_shape, in_channels, reuse=False):
    with tf.variable_scope('up_convolution', reuse=reuse):
        x = deconvolution_2d(layer_input, [2, 2, in_channels // 2, in_channels], output_shape, [1, 2, 2, 1])
        return prelu(x)


def up_convolution_resize(layer_input, output_shape, in_channels, factor=2, reuse=False):
    with tf.variable_scope('up_convolution', reuse=reuse):
        batch_list = [None]*layer_input.get_shape().as_list()[0]
        for i in range(int(layer_input.get_shape()[0])):
            channel_list = [None]*layer_input.get_shape().as_list()[3]
            for ii in range(int(layer_input.get_shape()[3])):
                channel_list[ii] = tf.image.resize_images(tf.expand_dims(tf.expand_dims(layer_input[i,:,:,ii],axis=0),axis=3), [int(output_shape[1]),int(output_shape[2])],method=0)
            batch_list[i] = tf.concat(channel_list,axis=3)
        input_up = tf.concat(batch_list,axis=0)
        x = convolution_2d(input_up, [3, 3, in_channels, in_channels//2], [1, 1, 1, 1])
        return prelu(x)
    
    
def resize_by_axis(image, dim_1, dim_2, ax, is_grayscale):
    resized_list = []
    if is_grayscale:
        unstack_img_depth_list = [tf.expand_dims(x,2) for x in tf.unstack(image, axis = ax)]
        for i in unstack_img_depth_list:
            resized_list.append(tf.image.resize_images(i, [dim_1, dim_2],method=0))
        stack_img = tf.squeeze(tf.stack(resized_list, axis=ax))
    else:
        unstack_img_depth_list = tf.unstack(image, axis = ax)
        for i in unstack_img_depth_list:
            resized_list.append(tf.image.resize_images(i, [dim_1, dim_2],method=0))
        stack_img = tf.stack(resized_list, axis=ax)
    return stack_img


def resize_by_axis_2d(image, dim_1, dim_2, ax, is_grayscale):
    resized_list = []
    if is_grayscale:
        unstack_img_depth_list = [tf.expand_dims(x,2) for x in tf.unstack(image, axis = ax)]
        for i in unstack_img_depth_list:
            resized_list.append(tf.image.resize_images(i, [dim_1, dim_2],method=0))
        stack_img = tf.squeeze(tf.stack(resized_list, axis=ax))
    else:
        unstack_img_depth_list = tf.unstack(image, axis = ax)
        for i in unstack_img_depth_list:
            resized_list.append(tf.image.resize_images(i, [dim_1, dim_2],method=0))
        stack_img = tf.stack(resized_list, axis=ax)
    return stack_img


def fcn_encoder(tf_input, input_channels, output_channels=1, n_channels=2, reuse=False):
    with tf.variable_scope('contracting_path',reuse=reuse):

        # if the input has more than 1 channel it has to be expanded because broadcasting only works for 1 input channel
        if input_channels == 1:
            c0 = tf.tile(tf_input, [1, 1, 1, n_channels])
        else:
            with tf.variable_scope('level_0',reuse=reuse):
                c0 = prelu(convolution_2d(tf_input, [5, 5, input_channels, n_channels], [1, 1, 1, 1]))
        with tf.variable_scope('level_1',reuse=reuse):
            c1 = convolution_block_in(c0, n_channels, 2, 5, 5,reuse=reuse)
            c12 = down_convolution(c1, n_channels,reuse=reuse)

        with tf.variable_scope('level_2',reuse=reuse):
            c2 = convolution_block_in(c12, n_channels * 2, 1, 5, 5,reuse=reuse)
            c22 = down_convolution(c2, n_channels * 2,reuse=reuse)

        with tf.variable_scope('level_3',reuse=reuse):
            c3 = convolution_block_in(c22, n_channels * 4, 2, 5, 5,reuse=reuse)
            c32 = down_convolution(c3, n_channels * 4,reuse=reuse)

        # with tf.variable_scope('level_4'):
        #    c4 = convolution_block_in(c32, n_channels * 8, 3, 5, 5, 5)
        #    c42 = down_convolution(c4, n_channels * 8)

        with tf.variable_scope('level_5',reuse=reuse):
            c5 = convolution_block_in(c32, n_channels * 8, 2, 5, 5,reuse=reuse)
    
    feature_list = [None]*4
    feature_list[0] = c1
    feature_list[1] = c2
    feature_list[2] = c3
    feature_list[3] = c5
    return feature_list


def fcn_encoder_batchnorm(tf_input, input_channels, output_channels=1, n_channels=2, reuse=False, training=True):
    with tf.variable_scope('contracting_path', reuse=reuse):

        # if the input has more than 1 channel it has to be expanded because broadcasting only works for 1 input channel
        if input_channels == 1:
            c0 = tf.tile(tf_input, [1, 1, 1, n_channels])
            c0 = tf.layers.batch_normalization(c0, momentum=0.99, epsilon=0.001, center=True, scale=True, training=training)
        else:
            with tf.variable_scope('level_0', reuse=reuse):
                c0 = tf.layers.batch_normalization(tf_input, momentum=0.99, epsilon=0.001, center=True, scale=True, training=training)
                c0 = convolution_2d(tf_input, [5, 5, input_channels, n_channels], [1, 1, 1, 1])
                c0 = prelu(c0)
                 
        with tf.variable_scope('level_1',reuse=reuse):
            c1 = convolution_block_in_batchnorm(c0, n_channels, 2, 5, 5, reuse=reuse, training=training)
            c12 = down_convolution(c1, n_channels, reuse=reuse)
            
        with tf.variable_scope('level_2',reuse=reuse):
            c2 = convolution_block_in_batchnorm(c12, n_channels * 2, 1, 5, 5, reuse=reuse, training=training)
            c22 = down_convolution(c2, n_channels * 2, reuse=reuse)
            
        with tf.variable_scope('level_3',reuse=reuse):
            c3 = convolution_block_in_batchnorm(c22, n_channels * 4, 2, 5, 5, reuse=reuse, training=training)
            c32 = down_convolution(c3, n_channels * 4, reuse=reuse)

        with tf.variable_scope('level_5',reuse=reuse):
            c5 = convolution_block_in_batchnorm(c32, n_channels * 8, 2, 5, 5, reuse=reuse, training=training)
    
    feature_list = [None]*4
    feature_list[0] = c1
    feature_list[1] = c2
    feature_list[2] = c3
    feature_list[3] = c5
    return feature_list


def fcn_decoder(merged_feature_list, output_channels=1, n_channels=2,reuse=False):
    c5 = merged_feature_list[3]
    c3 = merged_feature_list[2]
    c2 = merged_feature_list[1]
    c1 = merged_feature_list[0]
    with tf.variable_scope('contracting_path',reuse=reuse):
        with tf.variable_scope('level_5',reuse=reuse):
            c52 = up_convolution_resize(c5, c3.get_shape(), n_channels * 8, factor=2,reuse=reuse)

    with tf.variable_scope('expanding_path',reuse=reuse):

        # with tf.variable_scope('level_4'):
        #    e4 = convolution_block_2_in(c52, c4, n_channels * 8, 3, 5, 5, 5)
        #    e42 = up_convolution_resize(e4, c3.get_shape(), n_channels * 8, factor=2) # e42 = up_convolution(e4, tf.shape(c3), n_channels * 8)

        with tf.variable_scope('level_3',reuse=reuse):
            e3 = convolution_block_2_in(c52, c3, n_channels * 4, 2, 5, 5, reuse=reuse)
            e32 = up_convolution_resize(e3, c2.get_shape(), n_channels * 4, factor=2,reuse=reuse) # e32 = up_convolution(e3, tf.shape(c2), n_channels * 4)

        with tf.variable_scope('level_2',reuse=reuse):
            e2 = convolution_block_2_in(e32, c2, n_channels * 2, 1, 5, 5, reuse=reuse)
            e21 = up_convolution_resize(e2, c1.get_shape(), n_channels * 2, factor=2,reuse=reuse) # e21 = up_convolution(e2, tf.shape(c1), n_channels * 2)

        with tf.variable_scope('level_1',reuse=reuse):
            e1 = convolution_block_2_in(e21, c1, n_channels, 2, 5, 5, reuse=reuse)
            with tf.variable_scope('output_layer',reuse=reuse):
                logits = convolution_2d(e1, [1, 1, n_channels, output_channels], [1, 1, 1, 1])

    return logits


def fcn_decoder_batchnorm(merged_feature_list, output_channels=1, n_channels=2, reuse=False, training=True):
    c5 = merged_feature_list[3]
    c3 = merged_feature_list[2]
    c2 = merged_feature_list[1]
    c1 = merged_feature_list[0]
    
    with tf.variable_scope('contracting_path',reuse=reuse):
        with tf.variable_scope('level_5',reuse=reuse):
            c52 = up_convolution_resize(c5, c3.get_shape(), n_channels * 8, factor=2,reuse=reuse)
            
    with tf.variable_scope('expanding_path',reuse=reuse):

        with tf.variable_scope('level_3',reuse=reuse):
            e3 = convolution_block_2_in_batchnorm(c52, c3, n_channels * 4, 2, 5, 5, reuse=reuse, training=training)
            e32 = up_convolution_resize(e3, c2.get_shape(), n_channels * 4, factor=2,reuse=reuse)
        
        with tf.variable_scope('level_2',reuse=reuse):
            e2 = convolution_block_2_in_batchnorm(e32, c2, n_channels * 2, 1, 5, 5, reuse=reuse, training=training)
            e22 = up_convolution_resize(e2, c1.get_shape(), n_channels * 2, factor=2,reuse=reuse)
        
        with tf.variable_scope('level_1',reuse=reuse):
            e1 = convolution_block_2_in_batchnorm(e22, c1, n_channels, 2, 5, 5, reuse=reuse, training=training)
            with tf.variable_scope('output_layer',reuse=reuse):
                logits = convolution_2d(e1, [1, 1, n_channels, output_channels], [1, 1, 1, 1])

    return logits



def fcn_encoder_decoder(tf_input, input_channels, output_channels=1, n_channels=2,reuse=False):

    with tf.variable_scope('contracting_path',reuse=reuse):

        # if the input has more than 1 channel it has to be expanded because broadcasting only works for 1 input channel
        if input_channels == 1:
            c0 = tf.tile(tf_input, [1, 1, 1, n_channels])
        else:
            with tf.variable_scope('level_0',reuse=reuse):
                c0 = prelu(convolution_2d(tf_input, [5, 5, input_channels, n_channels], [1, 1, 1, 1]))

        with tf.variable_scope('level_1',reuse=reuse):
            c1 = convolution_block_in(c0, n_channels, 2, 5, 5,reuse=reuse)
            c12 = down_convolution(c1, n_channels,reuse=reuse)

        with tf.variable_scope('level_2',reuse=reuse):
            c2 = convolution_block_in(c12, n_channels * 2, 1, 5, 5,reuse=reuse)
            c22 = down_convolution(c2, n_channels * 2,reuse=reuse)

        with tf.variable_scope('level_3',reuse=reuse):
            c3 = convolution_block_in(c22, n_channels * 4, 2, 5, 5,reuse=reuse)
            c32 = down_convolution(c3, n_channels * 4,reuse=reuse)

        # with tf.variable_scope('level_4'):
        #    c4 = convolution_block_in(c32, n_channels * 8, 3, 5, 5, 5)
        #    c42 = down_convolution(c4, n_channels * 8)

        with tf.variable_scope('level_5',reuse=reuse):
            c5 = convolution_block_in(c32, n_channels * 8, 2, 5, 5,reuse=reuse)
            c52 = up_convolution_resize(c5, c3.get_shape(), n_channels * 8, factor=2,reuse=reuse)

    with tf.variable_scope('expanding_path',reuse=reuse):

        # with tf.variable_scope('level_4'):
        #    e4 = convolution_block_2_in(c52, c4, n_channels * 8, 3, 5, 5, 5)
        #    e42 = up_convolution_resize(e4, c3.get_shape(), n_channels * 8, factor=2) # e42 = up_convolution(e4, tf.shape(c3), n_channels * 8)

        with tf.variable_scope('level_3',reuse=reuse):
            e3 = convolution_block_2_in(c52, c3, n_channels * 4, 2, 5, 5,reuse=reuse)
            e32 = up_convolution_resize(e3, c2.get_shape(), n_channels * 4, factor=2,reuse=reuse) # e32 = up_convolution(e3, tf.shape(c2), n_channels * 4)

        with tf.variable_scope('level_2',reuse=reuse):
            e2 = convolution_block_2_in(e32, c2, n_channels * 2, 1, 5, 5,reuse=reuse)
            e22 = up_convolution_resize(e2, c1.get_shape(), n_channels * 2, factor=2,reuse=reuse) # e22 = up_convolution(e2, tf.shape(c1), n_channels * 2)

        with tf.variable_scope('level_1',reuse=reuse):
            e1 = convolution_block_2_in(e22, c1, n_channels, 2, 5, 5,reuse=reuse)
            with tf.variable_scope('output_layer',reuse=reuse):
                logits = convolution_2d(e1, [1, 1, n_channels, output_channels], [1, 1, 1, 1])

    return logits


def fcn_convGRU(feature_list,old_feature_list,reuse=False,format='list'):
    c5 = feature_list[3]
    c3 = feature_list[2]
    c2 = feature_list[1]
    c1 = feature_list[0]
    o1 = old_feature_list[0]
    o2 = old_feature_list[1]
    o3 = old_feature_list[2]
    o5 = old_feature_list[3]
    merged_feature_list = [None]*4
    with tf.variable_scope('merger', reuse=reuse):
        with tf.variable_scope('merge_layer_1', reuse=reuse):
            with tf.variable_scope('gru_gate_z', reuse=reuse):
                zt = convGRU_2d_gate(c1,o1,[5, 5, c1.get_shape()[3], c1.get_shape()[3]], [1, 1, 1, 1],padding='SAME')
            
            with tf.variable_scope('gru_gate_r', reuse=reuse):
                rt = convGRU_2d_gate(c1,o1,[5, 5, c1.get_shape()[3], c1.get_shape()[3]], [1, 1, 1, 1],padding='SAME')
            
            with tf.variable_scope('gru_output', reuse=reuse):
                merged_feature_list[0] = convGRU_2d_output(c1,o1,zt,rt,[5, 5, c1.get_shape()[3], c1.get_shape()[3]], [1, 1, 1, 1])
        
        with tf.variable_scope('merge_layer_2', reuse=reuse):
            with tf.variable_scope('gru_gate_z', reuse=reuse):
                zt = convGRU_2d_gate(c2,o2,[5, 5, c2.get_shape()[3], c2.get_shape()[3]], [1, 1, 1, 1],padding='SAME')
            
            with tf.variable_scope('gru_gate_r', reuse=reuse):
                rt = convGRU_2d_gate(c2,o2,[5, 5, c2.get_shape()[3], c2.get_shape()[3]], [1, 1, 1, 1],padding='SAME')
            
            with tf.variable_scope('gru_output', reuse=reuse):
                merged_feature_list[1] = convGRU_2d_output(c2,o2,zt,rt,[5, 5, c2.get_shape()[3], c2.get_shape()[3]], [1, 1, 1, 1])
        
        with tf.variable_scope('merge_layer_3', reuse=reuse):
            with tf.variable_scope('gru_gate_z', reuse=reuse):
                zt = convGRU_2d_gate(c3,o3,[5, 5, c3.get_shape()[3], c3.get_shape()[3]], [1, 1, 1, 1],padding='SAME')
            
            with tf.variable_scope('gru_gate_r', reuse=reuse):
                rt = convGRU_2d_gate(c3,o3,[5, 5, c3.get_shape()[3], c3.get_shape()[3]], [1, 1, 1, 1],padding='SAME')
            
            with tf.variable_scope('gru_output', reuse=reuse):
                merged_feature_list[2] = convGRU_2d_output(c3,o3,zt,rt,[5, 5, c3.get_shape()[3], c3.get_shape()[3]], [1, 1, 1, 1])
        
        with tf.variable_scope('merge_layer_4', reuse=reuse):
            with tf.variable_scope('gru_gate_z', reuse=reuse):
                zt = convGRU_2d_gate(c5,o5,[5, 5, c5.get_shape()[3], c5.get_shape()[3]], [1, 1, 1, 1],padding='SAME')
            
            with tf.variable_scope('gru_gate_r', reuse=reuse):
                rt = convGRU_2d_gate(c5,o5,[5, 5, c5.get_shape()[3], c5.get_shape()[3]], [1, 1, 1, 1],padding='SAME')
            
            with tf.variable_scope('gru_output', reuse=reuse):
                merged_feature_list[3] = convGRU_2d_output(c5,o5,zt,rt,[5, 5, c5.get_shape()[3], c5.get_shape()[3]], [1, 1, 1, 1])
    
    return merged_feature_list


def fcn_outer(x, c1, c2, c3, c5, reuse=tf.AUTO_REUSE, input_channels=2, output_channels=1, n_channels=2):
    print('FCN Model:')
    print('x: ' + str(x.get_shape()))
        
    logits_list = [None] * int(x.get_shape()[-1])
    for i in range(0, int(x.get_shape()[-1])):
        if tf.size(x) == 4:
            x_i = tf.expand_dims(x[:, :, :, i], axis=3)
        else:
            x_i = x[:, :, :, :, i]
        
        feature_list = fcn_encoder(x_i, input_channels=input_channels, n_channels=n_channels, reuse=tf.AUTO_REUSE)
        logits_list[i] = fcn_decoder(feature_list, output_channels=output_channels, n_channels=n_channels, reuse=tf.AUTO_REUSE)
        old_feature_list = feature_list.copy()
        
        print('logits_list: ' + str(logits_list[i].get_shape()))
        print('c1: ' + str(old_feature_list[0].get_shape()))
        print('c2: ' + str(old_feature_list[1].get_shape()))
        print('c3: ' + str(old_feature_list[2].get_shape()))
        print('c5: ' + str(old_feature_list[3].get_shape()))
        
    return (logits_list, old_feature_list)


def fcn_outer_batchnorm(x, c1, c2, c3, c5, reuse=tf.AUTO_REUSE, input_channels=2, output_channels=1, n_channels=2, training=True):
    print('FCN Model, with Batch Normalization:')
    print('x: ' + str(x.get_shape()))
        
    logits_list = [None] * int(x.get_shape()[-1])
    for i in range(0, int(x.get_shape()[-1])):
        if tf.size(x) == 4:
            x_i = tf.expand_dims(x[:, :, :, i], axis=3)
        else:
            x_i = x[:, :, :, :, i]
        
        feature_list = fcn_encoder_batchnorm(x_i, input_channels=input_channels, n_channels=n_channels, reuse=tf.AUTO_REUSE, training=training)
        logits_list[i] = fcn_decoder_batchnorm(feature_list, output_channels=output_channels, n_channels=n_channels, reuse=tf.AUTO_REUSE, training=training)
        old_feature_list = feature_list.copy()
        
        print('logits_list: ' + str(logits_list[i].get_shape()))
        print('c1: ' + str(old_feature_list[0].get_shape()))
        print('c2: ' + str(old_feature_list[1].get_shape()))
        print('c3: ' + str(old_feature_list[2].get_shape()))
        print('c5: ' + str(old_feature_list[3].get_shape()))
        
    return (logits_list, old_feature_list)


def rfcn_outer(x, c1, c2, c3, c5, reuse=tf.AUTO_REUSE, input_channels=1, output_channels=1, n_channels=2):
    print('RFCN Model CInput:')
    print('x: ' + str(x.get_shape()))
    
    logits_list = [None] * int(x.get_shape()[-1])
    for i in range(0, int(x.get_shape()[-1])):
        if tf.size(x) == 4:
            x_i = tf.expand_dims(x[:, :, :, i], axis=3)
        else:
            x_i = x[:, :, :, :, i]
        
        feature_list = fcn_encoder(x_i, input_channels=input_channels, n_channels=n_channels, reuse=tf.AUTO_REUSE)

        with tf.variable_scope('rnn', reuse=tf.AUTO_REUSE):
            if i == 0:  
                old_feature_list = [None]*len(feature_list)
                old_feature_list[0] = c1
                old_feature_list[1] = c2
                old_feature_list[2] = c3
                old_feature_list[3] = c5
            
            merged_feature_list = fcn_convGRU(feature_list, old_feature_list)
            
        logits_list[i] = fcn_decoder(merged_feature_list, output_channels=output_channels, n_channels=n_channels, reuse=tf.AUTO_REUSE)
        old_feature_list = merged_feature_list.copy()
        
        print('logits_list: ' + str(logits_list[i].get_shape()))
        print('c1: ' + str(old_feature_list[0].get_shape()))
        print('c2: ' + str(old_feature_list[1].get_shape()))
        print('c3: ' + str(old_feature_list[2].get_shape()))
        print('c5: ' + str(old_feature_list[3].get_shape()))
        
    return (logits_list, old_feature_list)


def rfcn_outer_batchnorm(x, c1, c2, c3, c5, reuse=tf.AUTO_REUSE, input_channels=1, output_channels=1, n_channels=2, training=True):
    print('RFCN Model CInput:')
    print('x: ' + str(x.get_shape()))
    
    logits_list = [None] * int(x.get_shape()[-1])
    for i in range(0, int(x.get_shape()[-1])):
        if tf.size(x) == 4:
            x_i = tf.expand_dims(x[:, :, :, i], axis=3)
        else:
            x_i = x[:, :, :, :, i]
        
        feature_list = fcn_encoder_batchnorm(x_i, input_channels=input_channels, n_channels=n_channels, reuse=tf.AUTO_REUSE, training=training)

        with tf.variable_scope('rnn', reuse=tf.AUTO_REUSE):
            if i == 0:  
                old_feature_list = [None]*len(feature_list)
                old_feature_list[0] = c1
                old_feature_list[1] = c2
                old_feature_list[2] = c3
                old_feature_list[3] = c5
            
            merged_feature_list = fcn_convGRU(feature_list, old_feature_list)
            
        logits_list[i] = fcn_decoder_batchnorm(merged_feature_list, output_channels=output_channels, n_channels=n_channels, reuse=tf.AUTO_REUSE, training=training)
        old_feature_list = merged_feature_list.copy()
        
        print('logits_list: ' + str(logits_list[i].get_shape()))
        print('c1: ' + str(old_feature_list[0].get_shape()))
        print('c2: ' + str(old_feature_list[1].get_shape()))
        print('c3: ' + str(old_feature_list[2].get_shape()))
        print('c5: ' + str(old_feature_list[3].get_shape()))
        
    return (logits_list, old_feature_list)