#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Example Parameters module
"""

#from collections import OrderedDict
from bson import SON

import scipy as sp

import cairo_objects 

import math

model = SON([

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
                     ('sx',.5),
                     ('sy',2),
                     ('object', [SON([('type','arc'),('args',(0,0,math.sqrt(.3*.3 / math.pi),.0,2*math.pi))])]),
                     ('pattern', SON([('args',[1, 0, 0]), ('type', 'SolidPattern')])),
                     ]),SON([('generator', 'cairo'),
                     ('height', 64),
                     ('width',64),
                     ('sx',1/1.8),
                     ('sy',1.8),
                     ('object', [SON([('type','arc'),('args',(0,0,math.sqrt(.3*.3 / math.pi),.0,2*math.pi))])]),
                     ('pattern', SON([('args',[1, 0, 0]), ('type', 'SolidPattern')])),
                     ])
                     
                     ]
    )])
),


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


])



# dict with all representation parameters
config = {

'models':[model],

'image' : SON([
      ('generator' , 'cairo'),
      ('width' , 64),
      ('height' , 64),
      ('objects' , [[SON([('type','arc'),('args',(0,0,math.sqrt(.3*.3 / math.pi),0,2*math.pi))])]]),
      ('patterns' , [cairo_objects.SOLID_RED]),
      ('sx' , SON([('$gt' , .5) , ('$lt' , 2.01) , ('delta' , .5)])),
      ('sy' , SON([('$gt' , .5) , ('$lt' , 2.01) , ('delta' , .5)])),
      ('tx' , SON([('$gt' , -.3) , ('$lt' , .301) , ('delta' , .03)])),
      ('ty' , SON([('$gt' , -.3) , ('$lt' , .301) , ('delta' , .03)]))
   ]) 


,

'train_test' : [
   SON([
      ('N',10), 
      ('ntrain',32),
      ('ntest',32),
      ('universe',SON([('image.tx',SON([('$lt',.25),('$gt',-.25)])),('image.ty',SON([('$lt',.25),('$gt',-.25)])),('$or',[SON([('image.sx',1),('image.sy',1)]),SON([('image.sx',.5),('image.sy',2)])])])),
      ('query',SON([('image.sx',1),('image.sy',1)]))
   ]),

   SON([
      ('N',10), 
      ('ntrain',64),
      ('ntest',32),
      ('ntrain_pos',32),
      ('universe',SON([('image.tx',SON([('$lt',.25),('$gt',-.25)])),('image.ty',SON([('$lt',.25),('$gt',-.25)])),('$or',[SON([('image.sx',1),('image.sy',1)]),SON([('image.sx',.5),('image.sy',2)])])])),
      ('query',SON([('image.sx',1),('image.sy',1)]))
   ]),
   SON([
      ('N',10), 
      ('ntrain',128),
      ('ntest',32),
      ('ntrain_pos',64),
      ('universe',SON([('image.tx',SON([('$lt',.25),('$gt',-.25)])),('image.ty',SON([('$lt',.25),('$gt',-.25)])),('$or',[SON([('image.sx',1),('image.sy',1)]),SON([('image.sx',.5),('image.sy',2)])])])),
      ('query',SON([('image.sx',1),('image.sy',1)]))
   ]),    
   SON([
      ('N',10), 
      ('ntrain',128),
      ('ntest',32),
      ('ntrain_pos',64),
      ('universe',SON([('image.tx',SON([('$lt',.15),('$gt',-.15)])),('image.ty',SON([('$lt',.15),('$gt',-.15)])),('$or',[SON([('image.sx',1),('image.sy',1)]),SON([('image.sx',.5),('image.sy',2)])])])),
      ('query',SON([('image.sx',1),('image.sy',1)]))
   ]),       
   SON([
      ('N',10), 
      ('ntrain',128),
      ('ntest',32),
      ('ntrain_pos',64),
      ('universe',SON([('image.tx',SON([('$lt',.22),('$gt',-.22)])),('image.ty',SON([('$lt',.22),('$gt',-.22)])),('$or',[SON([('image.sx',1),('image.sy',1)]),SON([('image.sx',.5),('image.sy',2)])])])),
      ('query',SON([('image.sx',1),('image.sy',1)]))
   ]), 
   SON([
      ('N',10), 
      ('ntrain',256),
      ('ntest',32),
      ('ntrain_pos',128),
      ('universe',SON([('image.tx',SON([('$lt',.22),('$gt',-.22)])),('image.ty',SON([('$lt',.22),('$gt',-.22)])),('$or',[SON([('image.sx',1),('image.sy',1)]),SON([('image.sx',.5),('image.sy',2)])])])),
      ('query',SON([('image.sx',1),('image.sy',1)]))
   ]),    

   ]
}

