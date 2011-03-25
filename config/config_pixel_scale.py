#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" params for experiment showing that pixels can distinguish squares from 
    rectanges in a purely scale-invariant way. 
    it has a bunch of (non-aspect-ratio-preserving) scalings, with no
    translations or rotation, of a red square.
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
    # kernel size of the box low pass filter
    ('max_edge' , 150),
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
      ('sx' , SON([('$gt' , .1) , ('$lt' , 5) , ('delta' , .1)])),
      ('sy' , SON([('$gt' , .1) , ('$lt' , 5) , ('delta' , .1)])),
   ]) 


}


