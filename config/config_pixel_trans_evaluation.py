#!/usr/bin/env python
# -*- coding: utf-8 -*-
from bson import SON

# params for tasks that fill out training example curve and task difficult curve for pixel experiment 2
config = {
'train_test' : [
   ##various numbers of training examples
   SON([
      ('N',40), 
      ('ntrain',16),
      ('ntest',32),
      ('ntrain_pos',8),
      ('universe',SON([('$or',[SON([('image.sx',1),('image.sy',1)]),SON([('image.sx',.5),('image.sy',2)])])])),
      ('query',SON([('image.sx',1),('image.sy',1)]))
   ]),   
   SON([
      ('N',40), 
      ('ntrain',32),
      ('ntest',32),
      ('ntrain_pos',16),
      ('universe',SON([('$or',[SON([('image.sx',1),('image.sy',1)]),SON([('image.sx',.5),('image.sy',2)])])])),
      ('query',SON([('image.sx',1),('image.sy',1)]))
   ]),
   SON([
      ('N',40), 
      ('ntrain',64),
      ('ntest',32),
      ('ntrain_pos',32),
      ('universe',SON([('$or',[SON([('image.sx',1),('image.sy',1)]),SON([('image.sx',.5),('image.sy',2)])])])),
      ('query',SON([('image.sx',1),('image.sy',1)]))
   ]),
   SON([
      ('N',40), 
      ('ntrain',128),
      ('ntest',32),
      ('ntrain_pos',64),
      ('universe',SON([('$or',[SON([('image.sx',1),('image.sy',1)]),SON([('image.sx',.5),('image.sy',2)])])])),
      ('query',SON([('image.sx',1),('image.sy',1)]))
   ]),
   
   ###increasing difficulty of task by having more translation 
   SON([
      ('N',40), 
      ('ntrain',16),
      ('ntest',16),
      ('ntrain_pos',8),
      ('universe',SON([('image.tx',SON([('$gte',-.042),('$lte',.080)])),('image.ty',SON([('$gte',-.042),('$lte',.080)])),('$or',[SON([('image.sx',1),('image.sy',1)]),SON([('image.sx',.5),('image.sy',2)])])])),
      ('query',SON([('image.sx',1),('image.sy',1)]))
   ]),  
   SON([
      ('N',40), 
      ('ntrain',16),
      ('ntest',16),
      ('ntrain_pos',8),
      ('universe',SON([('image.tx',SON([('$gte',-.82),('$lte',.120)])),('image.ty',SON([('$gte',-.82),('$lte',.120)])),('$or',[SON([('image.sx',1),('image.sy',1)]),SON([('image.sx',.5),('image.sy',2)])])])),
      ('query',SON([('image.sx',1),('image.sy',1)]))
   ]),     
   SON([
      ('N',40), 
      ('ntrain',16),
      ('ntest',16),
      ('ntrain_pos',8),
      ('universe',SON([('image.tx',SON([('$gte',-.202),('$lte',.240)])),('image.ty',SON([('$gte',-.202),('$lte',.240)])),('$or',[SON([('image.sx',1),('image.sy',1)]),SON([('image.sx',.5),('image.sy',2)])])])),
      ('query',SON([('image.sx',1),('image.sy',1)]))
   ]),
   SON([
      ('N',40), 
      ('ntrain',16),
      ('ntest',16),
      ('ntrain_pos',8),
      ('universe',SON([('image.tx',SON([('$gte',-.282),('$lte',.320)])),('image.ty',SON([('$gte',-.282),('$lte',.320)])),('$or',[SON([('image.sx',1),('image.sy',1)]),SON([('image.sx',.5),('image.sy',2)])])])),
      ('query',SON([('image.sx',1),('image.sy',1)]))
   ]), 
   SON([
      ('N',40), 
      ('ntrain',256),
      ('ntest',32),
      ('ntrain_pos',128),
      ('universe',SON([('$or',[SON([('image.sx',1),('image.sy',1)]),SON([('image.sx',.5),('image.sy',2)])])])),
      ('query',SON([('image.sx',1),('image.sy',1)]))
   ]), 
   SON([
      ('N',40), 
      ('ntrain',400),
      ('ntest',32),
      ('ntrain_pos',200),
      ('universe',SON([('$or',[SON([('image.sx',1),('image.sy',1)]),SON([('image.sx',.5),('image.sy',2)])])])),
      ('query',SON([('image.sx',1),('image.sy',1)]))
   ]),    
   ]  
}

