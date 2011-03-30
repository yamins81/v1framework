#!/usr/bin/env python
# -*- coding: utf-8 -*-
from bson import SON

# params for tasks that fill out training example curve and task difficult curve for pixel experiment 2
config = {
'train_test' : [
   SON([
      ('N',40), 
      ('ntrain',62),
      ('ntest',32),
      ('ntrain_pos',32),
      ('universe',SON([('$or',[SON([('image.sx',1),('image.sy',1)]),SON([('image.sx',.5),('image.sy',2)])])])),
      ('query',SON([('image.sx',1),('image.sy',1)]))
   ]),

   ]  
}

