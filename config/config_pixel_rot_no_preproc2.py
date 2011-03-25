#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" params for experiment showing that pixels can distinguish squares from 
    rectanges in a purely rotation-invariant way.
    it has a small number of scalings, to produces a square a rectangle of equal
    area and a large number of rotations, of a red square.
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
      ('sx' , SON([('$gt' , .5) , ('$lt' , 2.1) , ('delta' , .25)])),
      ('sy' , SON([('$gt' , .5) , ('$lt' , 2.1) , ('delta' , .25)])),
      ('rxy' , SON([('$gt' , 0) , ('$lt' , math.pi/2) , ('delta' , math.pi/64)])),
   ]) 


}


