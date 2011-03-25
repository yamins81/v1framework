#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" params for experiment showing that pixels can't distinguish squares from 
    rectanges in a rotation-and-scale-invariant way.
    it has a bunch of scalings and rotations of a red square.
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
      ('rxy' , SON([('$gt' , 0) , ('$lt' , math.pi/2) , ('delta' , math.pi/32)])),
      ('sx' , SON([('$gt' , .5) , ('$lt' , 6.1) , ('delta' , .5)])),
      ('sy' , SON([('$gt' , .5) , ('$lt' , 6.1) , ('delta' , .5)])),
   ]) 


}


