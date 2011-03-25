#!/usr/bin/env python
# -*- coding: utf-8 -*-
from bson import SON

# params for tasks that fill out training example curve and task difficult curve for pixel experiment 4
# just increases in training example number, since multiple task difficulties are needed
config = {
'train_test' : [
   SON([
      ('N',60), 
      ('ntrain',30),
      ('ntest',20),
      ('ntrain_pos',15),
      ('universe',SON([('$or',[SON([('image.sx',.5),('image.sy',2)]),SON([('image.sx',1),('image.sy',1)])])])),
      ('query',SON([('image.sx',.5),('image.sy',2)]))
   ])   
   ] 
}

