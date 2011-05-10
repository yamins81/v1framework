#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Example Parameters module
"""

#from collections import OrderedDict

from bson import SON

config = {
    'images': SON([
      ('num_images',10), 
      ('generator' , 'renderman'),
      ('selection','random'),
      ('tx' , SON([('$gt' , -.25) , ('$lt' , .251)])),
      ('ty' , SON([('$gt' , -.25) , ('$lt' , .251)]))])

}
 





