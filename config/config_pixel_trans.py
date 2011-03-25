#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" params for pixel experiment showing that pixels are bad at
    distinguishing squares from rectangles in a translation invariant way
    it has a bunch of scalings of a square (non-aspect-ratio-preserving)
    and a bunch of translations of a red square.
"""

#from collections import OrderedDict
from bson import SON

import scipy as sp

import math

import cairo_objects 

pixel_model = SON([

#global
('color_space' , 'rgb'),
('conv_mode' , 'valid'),

# prepare images before processing
('preproc', SON([
    ('max_edge' , 150),
    # kernel size of the box low pass filter
    ('lsum_ksize' , 3),
    # how to resize the image
    ('resize_method' , 'bicubic'),            
    ])),


# - linear filtering
('filter', SON([
    ('model_name','pixels'),
    # kernel shape
    ('kshape' , [1,1]),
    ])),


])


# dict with all representation parameters
config = {

'models': [pixel_model],

'image' : SON([
      ('generator' , 'cairo'),
      ('width' , 128),
      ('height' , 128),
      ('objects' , [cairo_objects.SQUARE]),
      ('patterns' , [cairo_objects.SOLID_RED]),
      ('tx' , SON([('$gt' , -.4) , ('$lt' , .401) , ('delta' , .04)])),
      ('ty' , SON([('$gt' , -.4) , ('$lt' , .401) , ('delta' , .04)])),
      ('sx' , SON([('$gt' , .5) , ('$lt' , 2.1) , ('delta' , .5)])),
      ('sy' , SON([('$gt' , .5) , ('$lt' , 2.1) , ('delta' , .5)])),
   ]) 


}


