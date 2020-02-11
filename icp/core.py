#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tqdm import tqdm


def icp(source, destination, dim_offset='all', dim_scale='all', num_iter=100, use_gpu=True, disable_tqdm=False):
    """
    Tensorflow implementation of the iterative closest point procedure
    :param source: Nxd numpy array of source data points
    :param destination: Mxd numpy array of points to source points should be transformed to
    :param dim_offset: list of dimensions (0 to d - 1) for with an offset parameter shall be determined, if 'all' it is done for any dim
    :param dim_scale: list of dimensions (0 to d - 1) for with a scale parameter shall be determined, if 'all' it is done for any dim
    :param use_gpu: Flag, if true gpu is used for computations if available
    :param disable_tqdm: flag, it true no progress bar is shown during optimization
    """

    # Preprocessing
    d = destination.shape[1]
    if dim_offset == 'all':
        dim_offset = list(range(d))
    if dim_scale == 'all':
        dim_scale = list(range(d))
    if not isinstance(dim_scale, list):
        dim_scale = [dim_scale]
    if not isinstance(dim_offset, list):
        dim_offset = [dim_offset]

    # Find nearest destination point for each source point
    # Set up variables
    d = destination.shape[1]
    src = tf.Variable(source)
    dst = tf.constant(destination)
    scl = tf.Variable(np.ones(d))
    scl_new = tf.Variable(np.ones(d))
    off = tf.Variable(np.zeros(d))
    off_new = tf.Variable(np.zeros(d))
    dim_scl = tf.constant(dim_scale)
    dim_off = tf.constant(dim_offset)

    # Get distances
    src_squ = tf.reduce_sum(tf.square(source), axis=1, keepdims=True)
    dst_squ = tf.reduce_sum(tf.square(destination), axis=1, keepdims=True)
    dist = src_squ - 2 * tf.matmul(src, dst, transpose_b=True) + tf.transpose(dst_squ)
    ind_nn = tf.argmin(dist, axis=1)

    # Update parameter
    src_mean = tf.reduce_mean(src, axis=0, keepdims=True)
    src_gth = tf.gather(src, dim_scl, axis=1)
    dst_gth = tf.gather(dst, ind_nn, axis=0)
    dst_mean = tf.reduce_mean(dst_gth, axis=0, keepdims=True)
    dst_cnt = dst_gth - dst_mean
    update_scl_new = tf.scatter_update(scl_new, dim_scl, tf.tile((tf.reduce_sum(src_gth * tf.gather(dst_cnt, dim_scl, axis=1), keepdims=True)
                                                                  / tf.reduce_sum(src_gth * (src_gth - tf.gather(src_mean, dim_scl, axis=1)),
                                                                                  keepdims=True))[:, 0], [len(dim_scale)]))
    update_off_new = tf.scatter_update(off_new, dim_off, tf.gather(dst_mean[0, :], dim_offset)
                                       - tf.gather(scl_new, dim_offset) * tf.gather(src_mean[0, :], dim_offset))
    update_scl = tf.assign(scl, scl * scl_new)
    update_off = tf.assign(off, scl_new * off + off_new)
    update_src_scl = tf.assign(src, src * tf.expand_dims(scl_new, 0))
    update_src_off = tf.assign(src, src + tf.expand_dims(off_new, 0))

    # Do the optimization in a Tensorflow session
    if use_gpu:
        sess = tf.Session()
    else:
        sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
    sess.run(tf.global_variables_initializer())
    for _ in tqdm(range(num_iter), disable=disable_tqdm):
        sess.run(update_scl_new)
        sess.run(update_off_new)
        sess.run([update_scl, update_off])
        sess.run(update_src_scl)
        sess.run(update_src_off)

    return sess.run([scl, off])
