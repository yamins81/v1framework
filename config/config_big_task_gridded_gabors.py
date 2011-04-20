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
('filter',SON([
    ('model_name','gridded_gabor'),
    # kernel shape of the gabors
    ('kshape' , [43,43]),
    # list of orientations
    ('norients' , 16),
    # list of frequencies
    ('divfreqs' , [2, 3, 4, 6, 11, 18]),
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

config = {
     'models': [model],
     'image' : experiment(150,150,.05,.175*.175),
      
     'train_test' : [
		   SON([
		      ('transform_average', SON([('transform_name','translation')])), 
			  ('N',10), 
			  ('ntrain',128),
			  ('ntest',32),
			  ('ntrain_pos',64),
			  ('query',SON([('image.object.type','rectangle')]))
			  ]),     
		   SON([
		      ('transform_average', SON([('transform_name','translation')])), 
			  ('N',10), 
			  ('ntrain',128),
			  ('ntest',32),
			  ('ntrain_pos',64),
			  ('universe',SON([('image.sx',SON([('$lt',1.2),('$gt',.8)])),('image.sy',SON([('$lt',1.2),('$gt',.8)]))])),
			  ('query',SON([('image.object.type','rectangle')]))
			  ]), 
		   SON([
		      ('transform_average', SON([('transform_name','translation')])), 
			  ('N',10), 
			  ('ntrain',128),
			  ('ntest',32),
			  ('ntrain_pos',64),
			  ('universe',SON([('image.rxy',SON([('$lt',.45)]))])),
			  ('query',SON([('image.object.type','rectangle')]))
			  ]),
		   SON([
		      ('transform_average', SON([('transform_name','translation')])), 
			  ('N',10), 
			  ('ntrain',128),
			  ('ntest',32),
			  ('ntrain_pos',64),
			  ('universe',SON([('image.sx',SON([('$gt',1/1.075),('$lt',1.075)])),('image.sy',SON([('$gt',1/1.075),('$lt',1.075)]))])),
			  ('query',SON([('image.object.type','rectangle')]))
			  ]),
		   SON([
		      ('transform_average', SON([('transform_name','translation')])), 
			  ('N',10), 
			  ('ntrain',128),
			  ('ntest',32),
			  ('ntrain_pos',64),
			  ('ntest_pos',16),
			  ('universe',SON([('image.object.type','rectangle'),('$or',[SON([('image.sx',N(1,.05)),('image.sy',N(1,.05))]),
                                                      SON([('image.sx',N(1.7,.05)),('image.sy',N(1/1.7,.05))]),
                                                      SON([('image.sx',N(1/1.7,.05)),('image.sy',N(1.7,.05))])
                                                     ])])),
			  ('query',SON([('image.object.type','rectangle'),('image.sx',N(1,.05)),('image.sy',N(1,.05))]))
			  ]),			  
			  
     ]       
     
}
 





