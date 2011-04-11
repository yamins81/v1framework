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
('filter', SON([
    ('model_name','center_surround'),
    ('kshape' , [32,32]),
    # list of orientations
    ('base_images',[SON([('generator', 'cairo'),
                     ('height', 64),
                     ('width',64),
                     ('object', [SON([('args',[.075]),('type','set_line_width')]),SON([('args', [-0.2, -0.2, 0.4, 0.4]), ('type', 'rectangle')])]),
                     ('pattern', SON([('args',[1, 0, 0]), ('type', 'SolidPattern')])),
                     ('actions',['stroke'])
                     ]),
                     SON([('generator', 'cairo'),
                     ('height', 64),
                     ('width',64),
                     ('object', [SON([('args',[.075]),('type','set_line_width')]),SON([('args', [-0.2, -0.2, 0.4, 0.4]), ('type', 'rectangle')])]),
                     ('pattern', SON([('args',[1, 0, 0]), ('type', 'SolidPattern')])),
                     ('actions',['stroke_preserve','fill'])
                     ])]
    )
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


]),],

'image' : SON([
      ('generator' , 'cairo'),
      ('width' , 64),
      ('height' , 64),
      ('objects' , [[SON([('args',[.075]),('type','set_line_width')]),SON([('args', [-0.2, -0.2, 0.4, 0.4]), ('type', 'rectangle')])]]),
      ('patterns' , [cairo_objects.SOLID_RED]),
      ('action_lists', [['stroke'],['stroke_preserve','fill']]),
      ('tx' , SON([('$gt' , -.4) , ('$lt' , .401) , ('delta' , .02)])),
      ('ty' , SON([('$gt' , -.4) , ('$lt' , .401) , ('delta' , .02)]))
   ]) 


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
   SON([
      ('N',10), 
      ('ntrain',256),
      ('ntest',32),
      ('ntrain_pos',128),
      ('universe',SON([('image.tx',SON([('$lt',.22),('$gt',-.22)])),('image.ty',SON([('$lt',.22),('$gt',-.22)]))])),
      ('query',SON([('image.actions',['stroke'])]))
   ]),   

   ]
}

