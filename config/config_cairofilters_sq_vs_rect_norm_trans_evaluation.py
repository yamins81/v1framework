#!/usr/bin/env python
# -*- coding: utf-8 -*-
from bson import SON

# params for tasks that fill out training example curve and task difficult curve for pixel experiment 2
config = {
'train_test' : [
   SON([
      ('N',40), 
      ('ntrain',128),
      ('ntest',32),
      ('ntrain_pos',64),
      ('universe',SON([('$or',[SON([('image.sx',1),('image.sy',1)]),SON([('image.sx',.5),('image.sy',2)])])])),
      ('query',SON([('image.sx',1),('image.sy',1)]))
   ]),
   SON([
      ('N',10), 
      ('ntrain',1000),
      ('ntest',32),
      ('ntrain_pos',500),
      ('universe',SON([('image.tx',SON([('$lt',.25),('$gt',-.25)])),('image.ty',SON([('$lt',.25),('$gt',-.25)])),('$or',[SON([('image.sx',1),('image.sy',1)]),SON([('image.sx',.5),('image.sy',2)])])])),
      ('query',SON([('image.sx',1),('image.sy',1)]))
   ]),
   SON([
      ('N',10), 
      ('ntrain',700),
      ('ntest',32),
      ('ntrain_pos',350),
      ('universe',SON([('image.tx',SON([('$lt',.2),('$gt',-.2)])),('image.ty',SON([('$lt',.2),('$gt',-.2)])),('$or',[SON([('image.sx',1),('image.sy',1)]),SON([('image.sx',.5),('image.sy',2)])])])),
      ('query',SON([('image.sx',1),('image.sy',1)]))
   ])      

   ]  
}

