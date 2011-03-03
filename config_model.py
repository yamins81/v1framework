#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" V1-like(A) Parameters module

References:

How far can you get with a modern face recognition test set using only simple features?
IEEE Computer Vision and Pattern Recognition (CVPR 2009).
Pinto N, DiCarlo JJ, Cox DD

Establishing Good Benchmarks and Baselines for Face Recognition.
IEEE European Conference on Computer Vision (ECCV 2008)
Pinto N, DiCarlo JJ, Cox DD

Why is Real-World Visual Object Recognition Hard?
PLoS Computational Biology 4(1): e27 doi:10.1371/journal.pcbi.0040027 (2008)
Pinto N*, Cox DD*, DiCarlo JJ
 
"""

import scipy as sp


# dict with all representation parameters
config = {

'global' : {
   'color_space' : 'rgb',
   'conv_mode' : 'valid'
},
# - preprocessing
# prepare images before processing
'preproc': {
    # resize input images by keeping aspect ratio and fix the biggest edge
    'max_edge': 150,
    # kernel size of the box low pass filter
    'lsum_ksize': 3,
    # how to resize the image
    'resize_method': 'bicubic',            
    },

# - input local normalization
# local zero-mean, unit-magnitude
'normin': {
    # kernel shape of the local normalization
    'kshape': (3,3),
    # magnitude threshold
    # if the vector's length is below, it doesn't get resized
    'threshold': 1.0,
    },

# - linear filtering
'filter': {
    'model_name':'gridded_gabor',
    # kernel shape of the gabors
    'kshape': [43,43],
    # list of orientations
    'norients': 16,
    # list of frequencies
    'divfreqs': [2, 3, 4, 6, 11, 18],
    # list of phases
    'phases': [0],
    },

# - simple non-linear activation
'activ': {
    # minimum output (clamp)
    'minout': 0,
    # maximum output (clamp)
    'maxout': 1,
    },

# - output local normalization
'normout': {
    # kernel shape of the local normalization
    'kshape': (3,3),
    # magnitude threshold
    # if the vector's length is below, it doesn't get resized
    'threshold': 1.0,
    },

# - pooling
'pool': {
    # kernel size of the local sum (2d slice)
    'lsum_ksize': 17,
    # fixed output shape (only the first 2 dimensions, y and x)
    'outshape': (30,30),
    },

# -- featsel details what features you want to be included in the vector
'featsel' : {
    # Include representation output ? True or False
    'output': True,

    # Include grayscale values ? None or (height, width)    
    'input_gray': None,
    # Include color histograms ? None or nbins per color
    'input_colorhists': None,
    # Include input norm histograms ? None or (division, nfeatures)    
    'normin_hists': None,
    # Include filter output histograms ? None or (division, nfeatures)
    'filter_hists': None,
    # Include activation output histograms ? None or (division, nfeatures)    
    'activ_hists': None,
    # Include output norm histograms ? None or (division, nfeatures)
    'normout_hists': None,
    # Include representation output histograms ? None or (division, nfeatures)
    'pool_hists': None,
    },

'image' : {
      'model_ids':['MB30635','MB30634','MB30625','MB30335'],
      'tx':{'$gt':-6.5,'$lt':-1.5,'delta':.05},
      'ty':{'$gt':-2,'$lt':2 ,'delta':1},
      'tz':{'$gt':-2,'$lt':2,'delta':1},
      'rxy':0, 'rxz':0, 'ryz': 0
      
   } 

}

