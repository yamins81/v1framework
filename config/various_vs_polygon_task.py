from bson import SON

config = {


'train_test' : [

   #sq vs polygon
   
   SON([
      ('transform_average', SON([('transform_name','nothing')])),
      ('N',1), 
      ('ntrain',300),
      ('ntest',40),
      ('ntrain_pos',150),
      ('ntest_pos',20),
      ('universe',SON([('image.tx',SON([('$lt',.22),('$gt',-.22)])),('image.ty',SON([('$lt',.22),('$gt',-.22)]))])),
      ('query',SON([('image.object',SON([('$size',8)]))]))
   ]), 
   SON([
      ('transform_average', SON([('transform_name','nothing')])),
      ('N',1), 
      ('ntrain',300),
      ('ntest',40),
      ('ntrain_pos',150),
      ('ntest_pos',20),
      ('universe',SON([('image.tx',SON([('$lt',.22),('$gt',-.22)])),('image.ty',SON([('$lt',.22),('$gt',-.22)]))])),
      ('query',SON([('image.object.type','rectangle')]))
   ]), 
    SON([
      ('transform_average', SON([('transform_name','nothing')])),
      ('N',1), 
      ('ntrain',300),
      ('ntest',40),
      ('ntrain_pos',150),
      ('ntest_pos',20),
      ('universe',SON([('image.tx',SON([('$lt',.22),('$gt',-.22)])),('image.ty',SON([('$lt',.22),('$gt',-.22)]))])),
      ('query',SON([('image.object',SON([('$size',5)]))]))
   ]), 
   SON([
      ('transform_average', SON([('transform_name','nothing')])),
      ('N',1), 
      ('ntrain',128),
      ('ntest',32),
      ('ntrain_pos',64),
      ('ntest_pos',16),
      ('universe',SON([('image.tx',SON([('$lt',.22),('$gt',-.22)])),
                       ('image.ty',SON([('$lt',.22),('$gt',-.22)])),
                       ('image.pattern.args',[1,0,0,1.0])])),
      ('query',SON([('image.object',SON([('$size',8)]))]))
   ]), 

   SON([
      ('transform_average', SON([('transform_name','nothing')])),
      ('N',1), 
      ('ntrain',128),
      ('ntest',32),
      ('ntrain_pos',64),
      ('ntest_pos',16),
      ('universe',SON([('image.tx',SON([('$lt',.22),('$gt',-.22)])),
                       ('image.ty',SON([('$lt',.22),('$gt',-.22)])),
                       ('image.pattern.args',[1,0,0,1.0])])),
      ('query',SON([('image.object',SON([('$size',5)]))]))
   ]),
   SON([
      ('transform_average', SON([('transform_name','nothing')])),
      ('N',1), 
      ('ntrain',128),
      ('ntest',32),
      ('ntrain_pos',64),
      ('ntest_pos',16),
      ('universe',SON([('image.tx',SON([('$lt',.22),('$gt',-.22)])),
                       ('image.ty',SON([('$lt',.22),('$gt',-.22)])),
                       ('$where' , '(this.config.image.sx / this.config.image.sy < 1.1) && (this.config.image.sx / this.config.image.sy > 1/1.1)')])),
      ('query',SON([('image.object',SON([('$size',5)]))]))
   ]),
   SON([
      ('transform_average', SON([('transform_name','nothing')])),
      ('N',1), 
      ('ntrain',128),
      ('ntest',32),
      ('ntrain_pos',64),
      ('ntest_pos',16),
      ('universe',SON([('$or',[SON([('image.object',SON([('$size',8)]))]),SON([('image.object',SON([('$size',7)]))]),SON([('image.object.type','rectangle')]),SON([('image.object.type','arc')])]),
                       ('image.tx',SON([('$lt',.22),('$gt',-.22)])),
                       ('image.ty',SON([('$lt',.22),('$gt',-.22)]))])),
      ('query',SON([('image.object',SON([('$size',8)]))]))
   ]), 
   SON([
      ('transform_average', SON([('transform_name','nothing')])),
      ('N',1), 
      ('ntrain',128),
      ('ntest',32),
      ('ntrain_pos',64),
      ('ntest_pos',16),
      ('universe',SON([('$or',[SON([('image.object',SON([('$size',8)]))]),SON([('image.object',SON([('$size',7)]))]),SON([('image.object.type','rectangle')])]),
                       ('image.tx',SON([('$lt',.22),('$gt',-.22)])),
                       ('image.ty',SON([('$lt',.22),('$gt',-.22)]))])),
      ('query',SON([('image.object',SON([('$size',8)]))]))
   ]),
   SON([
      ('transform_average', SON([('transform_name','nothing')])),
      ('N',1), 
      ('ntrain',128),
      ('ntest',32),
      ('ntrain_pos',64),
      ('ntest_pos',16),
      ('universe',SON([('$or',[SON([('image.object',SON([('$size',8)]))]),SON([('image.object',SON([('$size',7)]))]),SON([('image.object.type','rectangle')]),SON([('image.object.type','arc')])]),
                       ('image.tx',SON([('$lt',.22),('$gt',-.22)])),
                       ('image.ty',SON([('$lt',.22),('$gt',-.22)])),
                       ('image.pattern.args',[1,0,0,1.0])])),
      ('query',SON([('image.object',SON([('$size',8)]))]))
   ]),
   SON([
      ('transform_average', SON([('transform_name','nothing')])),
      ('N',1), 
      ('ntrain',128),
      ('ntest',32),
      ('ntrain_pos',64),
      ('ntest_pos',16),
      ('universe',SON([('$or',[SON([('image.object',SON([('$size',8)]))]),SON([('image.object',SON([('$size',7)]))]),SON([('image.object.type','rectangle')]),SON([('image.object.type','arc')])]),
                       ('image.tx',SON([('$lt',.22),('$gt',-.22)])),
                       ('image.ty',SON([('$lt',.22),('$gt',-.22)])),
                       ('$where' , '(this.config.image.sx / this.config.image.sy < 1.1) && (this.config.image.sx / this.config.image.sy > 1/1.1)')])),
      ('query',SON([('image.object',SON([('$size',8)]))]))
   ]),
   SON([
      ('transform_average', SON([('transform_name','nothing')])),
      ('N',1), 
      ('ntrain',128),
      ('ntest',32),
      ('ntrain_pos',64),
      ('ntest_pos',16),
      ('universe',SON([('$or',[SON([('image.object',SON([('$size',8)]))]),SON([('image.object',SON([('$size',7)]))]),SON([('image.object.type','rectangle')]),SON([('image.object.type','arc')])]),
                       ('image.tx',SON([('$lt',.22),('$gt',-.22)])),
                       ('image.ty',SON([('$lt',.22),('$gt',-.22)])),
                       ('$where' , '(this.config.image.sx / this.config.image.sy < 1.5) && (this.config.image.sx / this.config.image.sy > 1/1.5)')])),
      ('query',SON([('image.object',SON([('$size',8)]))]))
   ]),
   SON([
      ('transform_average', SON([('transform_name','nothing')])),
      ('N',1), 
      ('ntrain',128),
      ('ntest',32),
      ('ntrain_pos',64),
      ('ntest_pos',16),
      ('universe',SON([('image.tx',SON([('$lt',.22),('$gt',-.22)])),
                       ('image.ty',SON([('$lt',.22),('$gt',-.22)])),
                       ('$where' , '(this.config.image.sx / this.config.image.sy < 1.5) && (this.config.image.sx / this.config.image.sy > 1/1.5)')])),
      ('query',SON([('image.object',SON([('$size',8)]))]))
   ]),
   SON([
      ('transform_average', SON([('transform_name','nothing')])),
      ('N',1), 
      ('ntrain',128),
      ('ntest',40),
      ('ntrain_pos',64),
      ('ntest_pos',20),
      ('universe',SON([('$or',[SON([('image.object',SON([('$size',8)]))]),SON([('image.object',SON([('$size',5)]))])]),
                       ('image.tx',SON([('$lt',.22),('$gt',-.22)])),
                       ('image.ty',SON([('$lt',.22),('$gt',-.22)])),
                       ('$where' , '(this.config.image.sx / this.config.image.sy < 1.5) && (this.config.image.sx / this.config.image.sy > 1/1.5)')])),
      ('query',SON([('image.object',SON([('$size',8)]))]))
   ]),
   SON([
      ('transform_average', SON([('transform_name','nothing')])),
      ('N',1), 
      ('ntrain',128),
      ('ntest',40),
      ('ntrain_pos',64),
      ('ntest_pos',20),
      ('universe',SON([('$or',[SON([('image.object',SON([('$size',8)]))]),SON([('image.object',SON([('$size',5)]))])]),
                       ('image.tx',SON([('$lt',.22),('$gt',-.22)])),
                       ('image.ty',SON([('$lt',.22),('$gt',-.22)])),
                       ('$where' , '(this.config.image.sx / this.config.image.sy < 1.1) && (this.config.image.sx / this.config.image.sy > 1/1.1)')])),
      ('query',SON([('image.object',SON([('$size',8)]))]))
   ]),
   SON([
      ('transform_average', SON([('transform_name','nothing')])),
      ('N',1), 
      ('ntrain',150),
      ('ntest',32),
      ('ntrain_pos',75),
      ('ntest_pos',16),
      ('universe',SON([('image.tx',SON([('$lt',.22),('$gt',-.22)])),
                       ('image.ty',SON([('$lt',.22),('$gt',-.22)])),
                       ('$where' , '(this.config.image.sx / this.config.image.sy < 2.0) && (this.config.image.sx / this.config.image.sy > 1/2.0)')])),
      ('query',SON([('image.object',SON([('$size',8)]))]))
   ]),
   SON([
      ('transform_average', SON([('transform_name','nothing')])),
      ('N',1), 
      ('ntrain',150),
      ('ntest',32),
      ('ntrain_pos',75),
      ('ntest_pos',16),
      ('universe',SON([('image.tx',SON([('$lt',.22),('$gt',-.22)])),
                       ('image.ty',SON([('$lt',.22),('$gt',-.22)])),
                       ('$where' , '(this.config.image.sx / this.config.image.sy < 2.0) && (this.config.image.sx / this.config.image.sy > 1/2.0)')])),
      ('query',SON([('image.object',SON([('$size',8)]))]))
   ]),
   SON([
      ('transform_average', SON([('transform_name','nothing')])),
      ('N',1), 
      ('ntrain',150),
      ('ntest',32),
      ('ntrain_pos',75),
      ('ntest_pos',16),
      ('universe',SON([('$or',[SON([('image.object',SON([('$size',8)]))]),SON([('image.object',SON([('$size',7)]))]),SON([('image.object.type','rectangle')]),SON([('image.object.type','arc')])]),
                       ('image.tx',SON([('$lt',.22),('$gt',-.22)])),
                       ('image.ty',SON([('$lt',.22),('$gt',-.22)])),
                       ('$where' , '(this.config.image.sx / this.config.image.sy < 2.0) && (this.config.image.sx / this.config.image.sy > 1/2.0)')])),
      ('query',SON([('image.object',SON([('$size',8)]))]))
   ]),
   
   SON([
      ('transform_average', SON([('transform_name','nothing')])),
      ('N',1), 
      ('ntrain',300),
      ('ntest',40),
      ('universe',SON([('image.tx',SON([('$lt',.22),('$gt',-.22)])),
                       ('image.ty',SON([('$lt',.22),('$gt',-.22)])),
                       ('$where' , '(this.config.image.sx / this.config.image.sy < 1.1) && (this.config.image.sx / this.config.image.sy > 1/1.1)')])),
      ('query',[SON([('image.object.type','rectangle')]),SON([('image.object.type','arc')])])
   ]),
   SON([
      ('transform_average', SON([('transform_name','nothing')])),
      ('N',1), 
      ('ntrain',600),
      ('ntest',60),
      ('universe',SON([('image.tx',SON([('$lt',.22),('$gt',-.22)])),
                       ('image.ty',SON([('$lt',.22),('$gt',-.22)])),
                       ('$where' , '(this.config.image.sx / this.config.image.sy < 1.1) && (this.config.image.sx / this.config.image.sy > 1/1.1)')])),
      ('query',[SON([('image.object',SON([('$size',5)]))]),SON([('image.object.type','rectangle')]),SON([('image.object.type','arc')])])
   ]),
   SON([
      ('transform_average', SON([('transform_name','nothing')])),
      ('N',1), 
      ('ntrain',150),
      ('ntest',30),
      ('universe',SON([('image.tx',SON([('$lt',.22),('$gt',-.22)])),
                       ('image.ty',SON([('$lt',.22),('$gt',-.22)])),
                       ('$where' , '(this.config.image.sx / this.config.image.sy < 1.1) && (this.config.image.sx / this.config.image.sy > 1/1.1)')])),
      ('query',[SON([('image.object',SON([('$size',5)]))]),SON([('image.object.type','rectangle')]),SON([('image.object.type','arc')])])
   ]),
   SON([
      ('transform_average', SON([('transform_name','nothing')])),
      ('N',1), 
      ('ntrain',150),
      ('ntest',30),
      ('universe',SON([('image.tx',SON([('$lt',.22),('$gt',-.22)])),
                       ('image.ty',SON([('$lt',.22),('$gt',-.22)])),
                       ('$where' , '(this.config.image.sx / this.config.image.sy < 1.2) && (this.config.image.sx / this.config.image.sy > 1/1.2)')])),
      ('query',[SON([('image.object',SON([('$size',5)]))]),SON([('image.object.type','rectangle')]),SON([('image.object.type','arc')])])
   ]),
   SON([
      ('transform_average', SON([('transform_name','nothing')])),
      ('N',1), 
      ('ntrain',160),
      ('ntest',40),
      ('universe',SON([('image.tx',SON([('$lt',.22),('$gt',-.22)])),
                       ('image.ty',SON([('$lt',.22),('$gt',-.22)])),
                       ('$where' , '(this.config.image.sx / this.config.image.sy < 2.0) && (this.config.image.sx / this.config.image.sy > 1/2.0)')])),
      ('query',[SON([('image.object',SON([('$size',5)]))]),SON([('image.object',SON([('$size',7)]))]),SON([('image.object.type','rectangle')]),SON([('image.object.type','arc')])])
   ]),
   SON([
      ('transform_average', SON([('transform_name','nothing')])),
      ('N',1), 
      ('ntrain',100),
      ('ntest',40),
      ('universe',SON([('image.tx',SON([('$lt',.22),('$gt',-.22)])),
                       ('image.ty',SON([('$lt',.22),('$gt',-.22)])),
                       ('$where' , '(this.config.image.sx / this.config.image.sy < 2.0) && (this.config.image.sx / this.config.image.sy > 1/2.0)')])),
      ('query',[SON([('image.object',SON([('$size',8)]))]),SON([('image.object',SON([('$size',5)]))]),SON([('image.object',SON([('$size',7)]))]),SON([('image.object.type','rectangle')]),SON([('image.object.type','arc')])])
   ]),
   SON([
      ('transform_average', SON([('transform_name','nothing')])),
      ('N',1), 
      ('ntrain',80),
      ('ntest',32),
      ('universe',SON([('image.tx',SON([('$lt',.22),('$gt',-.22)])),
                       ('image.ty',SON([('$lt',.22),('$gt',-.22)])),
                       ('$where' , '(this.config.image.sx / this.config.image.sy < 2.0) && (this.config.image.sx / this.config.image.sy > 1/2.0)')])),
      ('query',[SON([('image.object',SON([('$size',5)]))]),SON([('image.object',SON([('$size',7)]))]),SON([('image.object.type','rectangle')]),SON([('image.object.type','arc')])])
   ]),
   SON([
      ('transform_average', SON([('transform_name','nothing')])),
      ('N',1), 
      ('ntrain',80),
      ('ntest',32),
      ('universe',SON([('image.tx',SON([('$lt',.22),('$gt',-.22)])),
                       ('image.ty',SON([('$lt',.22),('$gt',-.22)])),
                       ('$where' , '(this.config.image.sx / this.config.image.sy < 2.0) && (this.config.image.sx / this.config.image.sy > 1/2.0)')])),
      ('query',[SON([('image.object',SON([('$size',8)]))]),SON([('image.object',SON([('$size',7)]))]),SON([('image.object.type','rectangle')]),SON([('image.object.type','arc')])])
   ]),
   SON([
      ('transform_average', SON([('transform_name','nothing')])),
      ('N',1), 
      ('ntrain',200),
      ('ntest',60),
      ('universe',SON([('image.tx',SON([('$lt',.22),('$gt',-.22)])),
                       ('image.ty',SON([('$lt',.22),('$gt',-.22)])),
                       ('$where' , '(this.config.image.sx / this.config.image.sy < 2.0) && (this.config.image.sx / this.config.image.sy > 1/2.0)')])),
      ('query',[SON([('image.object',SON([('$size',8)]))]),SON([('image.object',SON([('$size',5)]))]),SON([('image.object',SON([('$size',7)]))]),SON([('image.object.type','rectangle')]),SON([('image.object.type','arc')])])
   ]),
   SON([
      ('transform_average', SON([('transform_name','nothing')])),
      ('N',1), 
      ('ntrain',160),
      ('ntest',48),
      ('universe',SON([('image.tx',SON([('$lt',.22),('$gt',-.22)])),
                       ('image.ty',SON([('$lt',.22),('$gt',-.22)])),
                       ('$where' , '(this.config.image.sx / this.config.image.sy < 2.0) && (this.config.image.sx / this.config.image.sy > 1/2.0)')])),
      ('query',[SON([('image.object',SON([('$size',8)]))]),SON([('image.object',SON([('$size',7)]))]),SON([('image.object.type','rectangle')]),SON([('image.object.type','arc')])])
   ]),
   SON([
      ('transform_average', SON([('transform_name','nothing')])),
      ('N',1), 
      ('ntrain',300),
      ('ntest',64),
      ('universe',SON([('image.tx',SON([('$lt',.22),('$gt',-.22)])),
                       ('image.ty',SON([('$lt',.22),('$gt',-.22)])),
                       ('$where' , '(this.config.image.sx / this.config.image.sy < 2.0) && (this.config.image.sx / this.config.image.sy > 1/2.0)')])),
      ('query',[SON([('image.object',SON([('$size',8)]))]),SON([('image.object',SON([('$size',7)]))]),SON([('image.object.type','rectangle')]),SON([('image.object.type','arc')])])
   ]),
   ]
   
   
}