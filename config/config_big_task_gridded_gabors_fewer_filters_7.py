#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Example Parameters module
"""

#from collections import OrderedDict

from bson import SON

def N(x,ep):
    from bson import SON
    return SON([('$gt',x-ep),('$lt',x + ep)])

def experiment(width,height,lw,A):
    
   import cairo_objects
   from bson import SON
   import math
   objects = [cairo_objects.square(area = A , lw = lw), 
              cairo_objects.disc(area = A, lw = lw),
              cairo_objects.polygon(3,area = A,lw=lw),
              cairo_objects.polygon(6,area = A, lw = lw),
             ]
              
   patterns = [cairo_objects.solid_surface(1,0,0),
               cairo_objects.solid_surface(1,1,0),
               cairo_objects.solid_surface(0,0,1),
               cairo_objects.linear_surface(0,0,1,1,[(.35,0,1,0,1),(.65,1,0,0,1)]),
               cairo_objects.radial_surface(.5,.5,0,.5,.5,2*math.sqrt(A/math.pi),[(.65,1,0,0,1),(.35,0,0,1,1)])
               ]
                         
   
   return SON([
      ('num_images',100000), 
      ('generator' , 'cairo'),
      ('width' , width),
      ('height' , height),
      ('objects' , objects),
      ('patterns' , patterns),
      ('action_lists', [['stroke'],['stroke_preserve','fill']]),
      ('tx' , SON([('$gt' , -.25) , ('$lt' , .251)])),
      ('ty' , SON([('$gt' , -.25) , ('$lt' , .251)])),
      ('sx' , SON([('$gt' , .5) , ('$lt' , 2.01)])),
      ('sy' , SON([('$gt' , .5) , ('$lt' , 2.01)])),
      ('rxy' , SON([('$gt' , 0) , ('$lt' , 2*math.pi)]))
      
   ]) 

import copy
base_model = SON([

#global
('color_space' , 'rgb'),
('conv_mode' , 'valid'),
           
# prepare images before processing
('preproc', SON([
    # resize input images by keeping aspect ratio and fix the biggest edge
    ('max_edge' , 64),
    # kernel size of the box low pass filter
    ('lsum_ksize' , None),
    # how to resize the image
    ('resize_method' , 'bicubic'), 
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
    # list of orientations
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
    
('flatten',False)    
])


models = []
#options = [(4,[2,3,4,5,6,7,9,10,12,14,16,18]),]
options = [(16,[4,5,6]),
           (16,[6,7,8]),
           (8,[4,5,6,7,8]),
           (4,[4,5,6,7,8]),
          ]          
kshapes = [[32,32]]          
          
for x in options:
    for kshape in kshapes:
        m = copy.deepcopy(base_model)
        m['filter']['norients'] = x[0]
        m['filter']['divfreqs'] = x[1]
        m['filter']['kshape'] = kshape
        models.append(m)
    
    
config = {
     'models': models,
     'image' : experiment(150,150,.05,.175*.175),
      
     'train_test' : [    
		   SON([
		      ('transform_average', SON([('transform_name','translation')])),
			  ('N',5), 
			  ('ntrain',128),
			  ('ntest',32),
			  ('ntrain_pos',64),
			  ('ntest_pos',32),
			  ('universe',SON([('image.rxy',SON([('$lt',.45)]))])),
			  ('query',SON([('image.object.type','rectangle')]))
			  ]),

     ]       
     
}
 





