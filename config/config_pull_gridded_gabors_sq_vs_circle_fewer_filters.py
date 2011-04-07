#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Example Parameters module
"""

#from collections import OrderedDict
from bson import SON

import scipy as sp

import cairo_objects 

import math

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
('normin', SON([
    # kernel shape of the local normalization
    ('kshape' , (3,3)),
    # magnitude threshold
    # if the vector's length is below, it doesn't get resized
    ('threshold' , 1.0),
    ])),

# - linear filtering
('filter',SON([
    ('model_name','gridded_gabor'),
    # kernel shape of the gabors
    ('kshape' , [32,32]),
    # list of orientations
    ('norients' , 8),
    # list of frequencies
    ('divfreqs' , [2, 6, 11, 18]),
    # list of phases
    ('phases' ,  [0]),
    ]) ),

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


]),],

'image' : SON([
      ('generator' , 'cairo'),
      ('width' , 64),
      ('height' , 64),
      ('objects' , [[SON([('type','rectangle'),('args',(-.2,-.2,.4,.4))])],[SON([('type','arc'),('args',(0,0,math.sqrt(.4*.4 / math.pi),.0,2*math.pi))])]]),
      ('patterns' , [cairo_objects.SOLID_RED]),
      ('tx' , SON([('$gt' , -.3) , ('$lt' , .301) , ('delta' , .02)])),
      ('ty' , SON([('$gt' , -.3) , ('$lt' , .301) , ('delta' , .02)]))
   ]) 


,

'train_test' : [

   SON([
      ('N',10), 
      ('ntrain',64),
      ('ntest',32),
      ('ntrain_pos',32),
      ('universe',SON([('image.tx',SON([('$lt',.25),('$gt',-.25)])),('image.ty',SON([('$lt',.25),('$gt',-.25)]))])),
      ('query',SON([('image.object.0.type','rectangle')]))
   ]),
   SON([
      ('N',10), 
      ('ntrain',128),
      ('ntest',32),
      ('ntrain_pos',64),
      ('universe',SON([('image.tx',SON([('$lt',.25),('$gt',-.25)])),('image.ty',SON([('$lt',.25),('$gt',-.25)]))])),
      ('query',SON([('image.object.0.type','rectangle')]))
   ]),      


   ]
}

