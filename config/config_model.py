#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Example Parameters module
"""

from collections import OrderedDict

import scipy as sp


# dict with all representation parameters
config = {

'global' : OrderedDict([
            ('color_space' , 'rgb'),
            ('conv_mode' , 'valid'),
           ]),
# - preprocessing
# prepare images before processing
'preproc': OrderedDict([
    # resize input images by keeping aspect ratio and fix the biggest edge
    ('max_edge' , 150),
    # kernel size of the box low pass filter
    ('lsum_ksize' , 3),
    # how to resize the image
    ('resize_method' , 'bicubic'),            
    ]),

# - input local normalization
# local zero-mean, unit-magnitude
'normin': OrderedDict([
    # kernel shape of the local normalization
    ('kshape' , (3,3)),
    # magnitude threshold
    # if the vector's length is below, it doesn't get resized
    ('threshold' , 1.0),
    ]),

# - linear filtering
'filter': OrderedDict([
    ('model_name','gridded_gabor'),
    # kernel shape of the gabors
    ('kshape' , [43,43]),
    # list of orientations
    ('norients' , 16),
    # list of frequencies
    ('divfreqs' , [2, 3, 4, 6, 11, 18]),
    # list of phases
    ('phases' ,  [0]),
    ]),

# - simple non-linear activation
'activ': OrderedDict([
    # minimum output (clamp)
    ('minout' , 0),
    # maximum output (clamp)
    ('maxout' , 1),
    ]),

# - output local normalization
'normout': OrderedDict([
    # kernel shape of the local normalization
    ('kshape', (3,3)),
    # magnitude threshold
    # if the vector's length is below, it doesn't get resized
    ('threshold', 1.0),
    ]),

# - pooling
'pool': OrderedDict([
    # kernel size of the local sum (2d slice)
    ('lsum_ksize' , 17 ),
    # fixed output shape (only the first 2 dimensions, y and x)
    ('outshape' , (30,30)),
    ]),

# -- featsel details what features you want to be included in the vector
'featsel' : OrderedDict([
    # Include representation output ? True or False
    ('output'  , True),
    # Include grayscale values ? None or (height, width)    
    ('input_gray' , None),
    # Include color histograms ? None or nbins per color
    ('input_colorhists' , None),
    # Include input norm histograms ? None or (division, nfeatures)    
    ('normin_hists' , None),
    # Include filter output histograms ? None or (division, nfeatures)
    ('filter_hists' , None),
    # Include activation output histograms ? None or (division, nfeatures)    
    ('activ_hists' , None),
    # Include output norm histograms ? None or (division, nfeatures)
    ('normout_hists' , None),
    # Include representation output histograms ? None or (division, nfeatures)
    ('pool_hists' , None),
    ]),

'image' : OrderedDict([
      ('generator' , 'renderman'), 
      ('model_ids',['MB30635','MB30634','MB30625','MB30335']),
      ('tx' , OrderedDict([('$gt',-6.5),('$lt',-1.5),('delta',.05)])),
      ('ty' , OrderedDict([('$gt',-2),('$lt',2),('delta',1)])),
      ('rxz', OrderedDict([('$gt',-3.14),('$lt',3.14),('delta',.5)])),
      
   ]) 

}

