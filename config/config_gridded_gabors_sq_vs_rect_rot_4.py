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
             ]
              
   patterns = [cairo_objects.solid_surface(1,0,0),
               ]
                         
   
   return SON([
      ('num_images',50000), 
      ('generator' , 'cairo'),
      ('width' , width),
      ('height' , height),
      ('objects' , objects),
      ('patterns' , patterns),
      ('tx' , SON([('$gt' , -.25) , ('$lt' , .251)])),
      ('ty' , SON([('$gt' , -.25) , ('$lt' , .251)])),
      ('sx' , SON([('$gt' , .5) , ('$lt' , 2.01) , ('delta',.5)])),
      ('sy' , SON([('$gt' , .5) , ('$lt' , 2.01) , ('delta',.5)])),
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

options = [
           (1,[2, 3, 4,6,10,12,14]),
           (2,[10])
          ]          
kshapes = [[20,20]]          
          
for x in options:
    for kshape in kshapes:
        m = copy.deepcopy(base_model)
        m['filter']['norients'] = x[0]
        m['filter']['divfreqs'] = x[1]
        m['filter']['kshape'] = kshape
        models.append(m)
    
    
config = {
     'models': models,
     'image' : experiment(64,64,.05,.175*.175),
      
     'train_test' : [  
 
       SON([
      ('transform_average', SON([('transform_name','translation')])),   
      ('N',5), 
      ('ntrain',250),
      ('ntest',32),
      ('ntrain_pos',125),
      ('ntest_pos',16),
      ('universe',SON([('image.rxy',SON([('$lt',.5)])),
                       ('image.tx',SON([('$lt',.25),('$gt',-.25)])),
                       ('image.ty',SON([('$lt',.25),('$gt',-.25)])),
                       ('$or',[SON([('image.sx',1),('image.sy',1)]),SON([('image.sx',.5),('image.sy',2)])])])),
      ('query',SON([('image.sx',1),('image.sy',1)]))
        ]),	
       SON([
      ('transform_average', SON([('transform_name','translation')])),   
      ('N',5), 
      ('ntrain',250),
      ('ntest',32),
      ('ntrain_pos',125),
      ('ntest_pos',16),
      ('universe',SON([('image.tx',SON([('$lt',.25),('$gt',-.25)])),
                       ('image.ty',SON([('$lt',.25),('$gt',-.25)])),
                       ('$or',[SON([('image.sx',1),('image.sy',1)]),SON([('image.sx',.5),('image.sy',2)])])])),
      ('query',SON([('image.sx',1),('image.sy',1)]))
        ]),	        
     ]       
     
}
 





