#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Example Parameters module
"""

#from collections import OrderedDict
from bson import SON

import scipy as sp

import cairo_objects 

import config_modifiers

import math

from dbutils import hgetattr, hsetattr

class OneGaborModifier(config_modifiers.BaseModifier):
    __module__ = 'config.config_greedy_optimization_onegabor_filterbank_sq_vs_rect' 
    def get_modifications(self,k,val):
        
        if k in ['filter.orients','filter.divfreqs','filter.phases']:    
            L = [val[0] + self.modifier_params[k]['delta'],val[0] - self.modifier_params[k]['delta'],val[0]]
            
            return [[l] for l in L if self.modifier_params[k]['min'] <= l <= self.modifier_params[k]['max']]
        
        elif k == 'filter.kshape':
            L = [val[0] + self.modifier_params[k]['delta'],val[0] - self.modifier_params[k]['delta'],val[0]]
            return [[l,l] for l in L if self.modifier_params[k]['min'] <= l <= self.modifier_params[k]['max']]
            
        else:
            raise ValueError, k + ' is not a recognized value'
         
    def get_vector(self,x0,x1,k):
        if k in ['filter.orients','filter.divfreqs','filter.phases','filter.kshape']:
            return 1 if (hgetattr(x1,k)[0] > hgetattr(x0,k)[0] ) else (-1 if (hgetattr(x1,k)[0] <  hgetattr(x0,k)[0]) else 0)
        else:
            raise ValueError, k + ' is not a recognized value'    
    
# dict with all representation parameters
config = SON([

('model', SON([

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
    ('model_name','specific_gabor'),
    # kernel shape of the gabors
    ('kshape' , [32,32]),
    # list of orientations
    ('orients' , [0]),
    # list of frequencies
    ('divfreqs' , [4]),
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


])),

('image' , SON([
      ('generator' , 'cairo'),
      ('width' , 64),
      ('height' , 64),
      ('objects' , [cairo_objects.SQUARE]),
      ('patterns' , [cairo_objects.SOLID_RED]),
      ('tx' , SON([('$gt' , -.4) , ('$lt' , .401) , ('delta' , .02)])),
      ('ty' , SON([('$gt' , -.4) , ('$lt' , .401) , ('delta' , .02)])),
      ('sx' , SON([('$gt' , .5) , ('$lt' , 2.1) , ('delta' , .5)])),
      ('sy' , SON([('$gt' , .5) , ('$lt' , 2.1) , ('delta' , .5)])),
   ]) 
)

,

('evaluation_task' , 
   SON([
      ('N',10), 
      ('ntrain',32),
      ('ntest',16),
      ('ntrain_pos',16),
      ('universe',SON([('image.tx',SON([('$lt',.25),('$gt',-.25)])),('image.ty',SON([('$lt',.25),('$gt',-.25)])),('$or',[SON([('image.sx',1),('image.sy',1)]),SON([('image.sx',.5),('image.sy',2)])])])),
      ('query',SON([('image.sx',1),('image.sy',1)]))
   ])
)
,

('modifier',OneGaborModifier),

('modifier_args',SON([('filter.orients', SON([('delta',.1),('min',0),('max',math.pi/2)])),
                      ('filter.divfreqs',SON([('delta',1),('min',2),('max',20)])),
                      ('filter.kshape', SON([('delta',1),('min',15),('max',40)])),
                     ])
),
('rep_limit',50)

])

