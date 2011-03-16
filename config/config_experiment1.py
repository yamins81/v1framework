#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Example Parameters module
"""

#from collections import OrderedDict
from bson import SON

import scipy as sp

import cairo_objects 

model1 = SON([

#global
('color_space' , 'rgb'),
('conv_mode' , 'valid'),
           
# prepare images before processing
('preproc', SON([
    # resize input images by keeping aspect ratio and fix the biggest edge
    ('max_edge' , 150),
    # kernel size of the box low pass filter
    ('lsum_ksize' , 3),
    # how to resize the image
    ('resize_method' , 'bicubic'),            
    ])),

# - input local normalization
# local zero-mean, unit-magnitude
('normin', SON([
    # kernel shape of the local normalization
    ('kshape' , (3,3)),
    # magnitude threshold
    # if the vector's length is below, it doesn't get resized
    ('threshold' , 1.0),
    ])),

# - linear filtering
('filter', SON([
    ('model_name','gridded_gabor'),
    # kernel shape of the gabors
    ('kshape' , [43,43]),
    # list of orientations
    ('norients' , 16),
    # list of frequencies
    ('divfreqs' , [2, 3, 4, 6, 11, 18]),
    # list of phases
    ('phases' ,  [0]),
    ])),

# - simple non-linear activation
('activ', SON([
    # minimum output (clamp)
    ('minout' , 0),
    # maximum output (clamp)
    ('maxout' , 1),
    ])),

# - output local normalization
('normout', SON([
    # kernel shape of the local normalization
    ('kshape', (3,3)),
    # magnitude threshold
    # if the vector's length is below, it doesn't get resized
    ('threshold', 1.0),
    ])),

# - pooling
('pool' , SON([
    # kernel size of the local sum (2d slice)
    ('lsum_ksize' , 17 ),
    # fixed output shape (only the first 2 dimensions, y and x)
    ('outshape' , (30,30)),
    ])),

# -- featsel details what features you want to be included in the vector
('featsel' , SON([
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
    ]))
])

model2 = model1.copy()
model2['filter'] = SON([
    ('model_name','random_gabor'),
    ('kshape' , [43,43]),
    ('max_wavelength',18),
    ('min_wavelength',2),
    ('num_filters',96),
    ])

model3 = model1.copy()
model3['filter'] = SON([
    ('model_name','really_random'),
    ('kshape' , [43,43]),
    ('num_filters' , 96),
    ])

# dict with all representation parameters
config = {

'models': [model1,model2,model3],

'image' : SON([
      ('generator' , 'cairo'),
      ('width' , 256),
      ('height' , 256),
      ('objects' , [cairo_objects.SQUARE]),
      ('patterns' , [cairo_objects.SOLID_RED]),
      ('tx' , SON([('$gt' , -.3) , ('$lt' , .3) , ('delta' , .1)])),
      ('ty' , SON([('$gt' , -.3) , ('$lt' , .3) , ('delta' , .1)])),
      ('sx' , SON([('$gt' , .5) , ('$lt' , 1.5) , ('delta' , .2)])),
      ('sy' , SON([('$gt' , .5) , ('$lt' , 1.5) , ('delta' , .2)])),
   ]) 


}


