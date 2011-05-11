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
      ('num_images',1000), 
      ('generator' , 'cairo'),
      ('selection','random'),
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


config = {
     'images' : experiment(150,150,.05,.175*.175),
}
 





