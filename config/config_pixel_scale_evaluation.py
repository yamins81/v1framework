#!/usr/bin/env python
# -*- coding: utf-8 -*-
from bson import SON

# params for tasks that fill out training example curve and task difficult curve for pixel experiment 3
config = {
'train_test' : [
   ###increasing numbers of training examples 
   SON([
      ('N',40), 
      ('ntrain',10),
      ('ntest',32),
      ('ntrain_pos',5),
      ('universe',SON([('$where','(((this.config.image.sx/this.config.image.sy >= 1.99) && (this.config.image.sx/this.config.image.sy <= 2.01)) || ((this.config.image.sx/this.config.image.sy >= .499) && (this.config.image.sx/this.config.image.sy <= .501)) || (this.config.image.sx/this.config.image.sy === 1.0))')])),
      ('query',SON([('$where','(this.config.image.sx/this.config.image.sy === 1.0)')]))
   ]),
   SON([
      ('N',40), 
      ('ntrain',20),
      ('ntest',32),
      ('ntrain_pos',10),
      ('universe',SON([('$where','(((this.config.image.sx/this.config.image.sy >= 1.99) && (this.config.image.sx/this.config.image.sy <= 2.01)) || ((this.config.image.sx/this.config.image.sy >= .499) && (this.config.image.sx/this.config.image.sy <= .501)) || (this.config.image.sx/this.config.image.sy === 1.0))')])),
      ('query',SON([('$where','(this.config.image.sx/this.config.image.sy === 1.0)')]))
   ]),  
   SON([
      ('N',40), 
      ('ntrain',30),
      ('ntest',32),
      ('ntrain_pos',15),
      ('universe',SON([('$where','(((this.config.image.sx/this.config.image.sy >= 1.99) && (this.config.image.sx/this.config.image.sy <= 2.01)) || ((this.config.image.sx/this.config.image.sy >= .499) && (this.config.image.sx/this.config.image.sy <= .501)) || (this.config.image.sx/this.config.image.sy === 1.0))')])),
      ('query',SON([('$where','(this.config.image.sx/this.config.image.sy === 1.0)')]))
   ]),    
   SON([
      ('N',40), 
      ('ntrain',60),
      ('ntest',32),
      ('ntrain_pos',30),
      ('universe',SON([('$where','(((this.config.image.sx/this.config.image.sy >= 1.99) && (this.config.image.sx/this.config.image.sy <= 2.01)) || ((this.config.image.sx/this.config.image.sy >= .499) && (this.config.image.sx/this.config.image.sy <= .501)) || (this.config.image.sx/this.config.image.sy === 1.0))')])),
      ('query',SON([('$where','(this.config.image.sx/this.config.image.sy === 1.0)')]))
   ]),    
   
   ###various task difficulties
   SON([
      ('N',40), 
      ('ntrain',80),
      ('ntest',15),
      ('ntrain_pos',40),
      ('universe',SON([('$where','(((this.config.image.sx/this.config.image.sy >= 1.99) && (this.config.image.sx/this.config.image.sy <= 2.01)) || ((this.config.image.sx/this.config.image.sy >= .499) && (this.config.image.sx/this.config.image.sy <= .501)) || (this.config.image.sx/this.config.image.sy === 1.0))')])),
      ('query',SON([('$where','(this.config.image.sx/this.config.image.sy === 1.0)')]))
   ]), 
   SON([
      ('N',60), 
      ('ntrain',128),
      ('ntest',64),
      ('ntrain_pos',64),
      ('universe',SON([('$where','(((this.config.image.sx/this.config.image.sy >= 1.99) && (this.config.image.sx/this.config.image.sy <= 3.01)) || ((this.config.image.sx/this.config.image.sy >= .333) && (this.config.image.sx/this.config.image.sy <= .501)) || ((this.config.image.sx/this.config.image.sy >= .8) && (this.config.image.sx/this.config.image.sy <= 1.2501)))')])),
      ('query',SON([('$where','((this.config.image.sx/this.config.image.sy >= .8) && (this.config.image.sx/this.config.image.sy <= 1.2501))')]))
   ]),
   SON([
      ('N',40), 
      ('ntrain',80),
      ('ntest',60),
      ('ntrain_pos',40),
      ('universe',SON([('$where','(((this.config.image.sx/this.config.image.sy >= 1.99) && (this.config.image.sx/this.config.image.sy <= 1.2)) || ((this.config.image.sx/this.config.image.sy >= .499) && (this.config.image.sx/this.config.image.sy <= 1/1.2)) || (this.config.image.sx/this.config.image.sy === 1.0))')])),
      ('query',SON([('$where','(this.config.image.sx/this.config.image.sy === 1.0)')]))
   ]),   
   ]  
}

