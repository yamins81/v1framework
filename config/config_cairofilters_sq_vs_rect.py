#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Example Parameters module
"""

#from collections import OrderedDict
from bson import SON

import scipy as sp

import cairo_objects 


# dict with all representation parameters
config = {

'models': [SON([

#global
('color_space' , 'rgb'),
('conv_mode' , 'valid'),
           
# prepare images before processing
('preproc', SON([
    # resize input images by keeping aspect ratio and fix the biggest edge
    ('max_edge' , None),
    # kernel size of the box low pass filter
    ('lsum_ksize' , None),
    # how to resize the image
    ('resize_method' , None), 
    ('whiten', False)
    ])),

# - input local normalization
# local zero-mean, unit-magnitude


# - linear filtering
('filter', SON([
    ('model_name','cairo_generated'),
    ('kshape' , [32,32]),
    # list of orientations
    ('specs',[SON([
      ('width',64),
      ('height',64),
      ('object' , cairo_objects.SQUARE),
      ('pattern' , cairo_objects.SOLID_RED),
      ]),SON([
      ('width',64),
      ('height',64),
      ('object' , cairo_objects.SQUARE),
      ('pattern' , cairo_objects.SOLID_RED),
      ('sx',.5),
      ('sy',2)
      ])] , )
    ])),

# - simple non-linear activation
('activ', SON([
    # minimum output (clamp)
    ('minout' , 0),
    # maximum output (clamp)
    ('maxout' , 1),
    ])),



]),],

'image' : SON([
      ('generator' , 'cairo'),
      ('width' , 64),
      ('height' , 64),
      ('objects' , [cairo_objects.SQUARE]),
      ('patterns' , [cairo_objects.SOLID_RED]),
      ('tx' , SON([('$gt' , -.2) , ('$lt' , .201) , ('delta' , .04)])),
      ('ty' , SON([('$gt' , -.2) , ('$lt' , .201) , ('delta' , .04)])),
      ('sx' , SON([('$gt' , .5) , ('$lt' , 2.1) , ('delta' , .5)])),
      ('sy' , SON([('$gt' , .5) , ('$lt' , 2.1) , ('delta' , .5)])),
   ]) 

}

