#!/usr/bin/env python
# -*- coding: utf-8 -*-
from bson import SON

# params for tasks that fill out training example curve and task difficult curve for pixel experiment 5
config = {
'train_test' : [
   ###a task it works on
   SON([
      ('N',60), 
      ('ntrain',64),
      ('ntest',32),
      ('ntrain_pos',32),
      ('universe',SON([('$or',[SON([('image.sx',t),('image.sy',t)]) for t in [1,2,3]] + [SON([('image.sx',t),('image.sy',4*t)]) for t in [.5,1,1.5]] + [SON([('image.sx',4*t),('image.sy',t)]) for t in [.5,1,1.5]])])),
      ('query',SON([('$or',[SON([('image.sx',t),('image.sy',t)]) for t in [1,2,3]])]))
   ]),  
   
   #difficult task
   SON([
      ('N',60), 
      ('ntrain',64),
      ('ntest',32),
      ('ntrain_pos',32),
      ('universe',SON([('image.sx',SON([('$lt',5)])),('image.sy',SON([('$lt',5)])),('$where','((this.config.image.sx/this.config.image.sy === 2.0) || (this.config.image.sx/this.config.image.sy === .50) || (this.config.image.sx/this.config.image.sy === 1.0))')])),
      ('query',SON([('$where','(this.config.image.sx/this.config.image.sy === 1.0)')]))
   ]),
   
   ###increasing difficulty by having more and more rotation
   SON([
      ('N',60), 
      ('ntrain',30),
      ('ntest',20),
      ('ntrain_pos',15),
      ('universe',SON([('image.rxy',SON([('$lt',.2)])),('image.sx',SON([('$lt',5)])),('image.sy',SON([('$lt',5)])),('$where','((this.config.image.sx/this.config.image.sy === 2.0) || (this.config.image.sx/this.config.image.sy === .50) || (this.config.image.sx/this.config.image.sy === 1.0))')])),
      ('query',SON([('$where','(this.config.image.sx/this.config.image.sy === 1.0)')]))
   ]) ,
   SON([
      ('N',60), 
      ('ntrain',30),
      ('ntest',20),
      ('ntrain_pos',15),
      ('universe',SON([('image.rxy',SON([('$lt',.4)])),('image.sx',SON([('$lt',5)])),('image.sy',SON([('$lt',5)])),('$where','((this.config.image.sx/this.config.image.sy === 2.0) || (this.config.image.sx/this.config.image.sy === .50) || (this.config.image.sx/this.config.image.sy === 1.0))')])),
      ('query',SON([('$where','(this.config.image.sx/this.config.image.sy === 1.0)')]))
   ]),
   SON([
      ('N',60), 
      ('ntrain',30),
      ('ntest',20),
      ('ntrain_pos',15),
      ('universe',SON([('image.rxy',SON([('$lt',.6)])),('image.sx',SON([('$lt',5)])),('image.sy',SON([('$lt',5)])),('$where','((this.config.image.sx/this.config.image.sy === 2.0) || (this.config.image.sx/this.config.image.sy === .50) || (this.config.image.sx/this.config.image.sy === 1.0))')])),
      ('query',SON([('$where','(this.config.image.sx/this.config.image.sy === 1.0)')]))
   ])  ,
   SON([
      ('N',60), 
      ('ntrain',30),
      ('ntest',20),
      ('ntrain_pos',15),
      ('universe',SON([('image.rxy',SON([('$lt',.8)])),('image.sx',SON([('$lt',5)])),('image.sy',SON([('$lt',5)])),('$where','((this.config.image.sx/this.config.image.sy === 2.0) || (this.config.image.sx/this.config.image.sy === .50) || (this.config.image.sx/this.config.image.sy === 1.0))')])),
      ('query',SON([('$where','(this.config.image.sx/this.config.image.sy === 1.0)')]))
   ])     
   ] 
}

