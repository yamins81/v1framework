#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Example Parameters module
"""

#from collections import OrderedDict
from bson import SON

import scipy as sp

import cairo_objects

import math
 
def experiment(width,height,lw,A):
       
   objects = [cairo_objects.square(area = A , lw = lw), 
   	          cairo_objects.disc(area = A, lw = lw),
   	          cairo_objects.polygon(3,area = A,lw=lw),
   	          cairo_objects.polygon(6,area = A, lw = lw),
   	          ]
   	          
   patterns = [cairo_objects.solid_surface(1,0,0),
               cairo_objects.solid_surface(0,0,1),
               cairo_objects.linear_surface(0,0,1,1,[(.35,0,1,0,1),(.65,1,0,0,1)]),
               cairo_objects.radial_surface(.5,.5,0,.5,.5,2*math.sqrt(A/math.pi),[(.65,1,0,0,1),(.35,0,0,1,1)])
               ]
                         
   
   return SON([
      ('generator' , 'cairo'),
      ('width' , width),
      ('height' , height),
      ('objects' , objects),
      ('patterns' , patterns),
      ('action_lists', [['stroke'],['stroke_preserve','fill']]),
      ('tx' , SON([('$gt' , -.3) , ('$lt' , .301) , ('delta' , .03)])),
      ('ty' , SON([('$gt' , -.3) , ('$lt' , .301) , ('delta' , .03)])),
      ('sx' , SON([('$gt' , .5) , ('$lt' , 2.01) , ('delta' , .5)])),
      ('sy' , SON([('$gt' , .5) , ('$lt' , 2.01) , ('delta' , .5)])),
      ('rxy' , SON([('$gt' , 0) , ('$lt' , 2*math.pi) , ('delta' , math.pi/3)]))
      
   ]) 

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
    ('norients' , 1),
    # list of frequencies
    ('divfreqs' , [2,  18]),
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

'image' : experiment(128,128,.05,.15*.15)
,

'train_test' : [


   SON([
      ('N',10), 
      ('ntrain',128),
      ('ntest',32),
      ('ntrain_pos',64),
      ('universe',SON([('image.tx',SON([('$lt',.22),('$gt',-.22)])),('image.ty',SON([('$lt',.22),('$gt',-.22)]))])),
      ('query',SON([('image.actions',['stroke'])]))
   ]),
   SON([
      ('N',10), 
      ('ntrain',64),
      ('ntest',32),
      ('ntrain_pos',32),
      ('universe',SON([('image.tx',SON([('$lt',.22),('$gt',-.22)])),('image.ty',SON([('$lt',.22),('$gt',-.22)]))])),
      ('query',SON([('image.actions',['stroke'])]))
   ]),
   SON([
      ('N',10), 
      ('ntrain',32),
      ('ntest',32),
      ('ntrain_pos',16),
      ('universe',SON([('image.tx',SON([('$lt',.22),('$gt',-.22)])),('image.ty',SON([('$lt',.22),('$gt',-.22)]))])),
      ('query',SON([('image.actions',['stroke'])]))
   ]),

   ]
}

