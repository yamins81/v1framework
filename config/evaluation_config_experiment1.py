#!/usr/bin/env python
# -*- coding: utf-8 -*-
from bson import SON

# dict with all representation parameters
config = {
'train_test' : [SON([
      ('ntrain',32),
      ('ntest',32),
      ('ntrain_pos',16),
      ('query',SON([('image.sx',SON([('$lte',1)]))]))
   ]),
   SON([
      ('ntrain',32),
      ('ntest',32),
      ('ntrain_pos',16),
      ('query',SON([('image.tx',SON([('$lte',0)]))]))
   ]),
   ]  
}







