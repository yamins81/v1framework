from bson import SON

config = {


'train_test' : [

   SON([
      ('transform_average', SON([('transform_name','translation')])),
      ('N',1), 
      ('ntrain',160),
      ('ntest',40),
      ('universe',SON([('image.tx',SON([('$lt',.22),('$gt',-.22)])),
                       ('image.ty',SON([('$lt',.22),('$gt',-.22)])),
                       ('$where' , '(this.config.image.sx / this.config.image.sy < 2.0) && (this.config.image.sx / this.config.image.sy > 1/2.0)')])),
      ('query',[SON([('image.object',SON([('$size',5)]))]),SON([('image.object',SON([('$size',7)]))]),SON([('image.object.type','rectangle')]),SON([('image.object.type','arc')])])
   ]),
   SON([
      ('transform_average', SON([('transform_name','translation'),('max',True)])),
      ('N',1), 
      ('ntrain',160),
      ('ntest',40),
      ('universe',SON([('image.tx',SON([('$lt',.22),('$gt',-.22)])),
                       ('image.ty',SON([('$lt',.22),('$gt',-.22)])),
                       ('$where' , '(this.config.image.sx / this.config.image.sy < 2.0) && (this.config.image.sx / this.config.image.sy > 1/2.0)')])),
      ('query',[SON([('image.object',SON([('$size',5)]))]),SON([('image.object',SON([('$size',7)]))]),SON([('image.object.type','rectangle')]),SON([('image.object.type','arc')])])
   ]),
   SON([
      ('transform_average', SON([('transform_name','translation')])),
      ('N',1), 
      ('ntrain',160),
      ('ntest',40),
      ('universe',SON([('image.tx',SON([('$lt',.22),('$gt',-.22)])),
                       ('image.ty',SON([('$lt',.22),('$gt',-.22)])),
                       ('image.rxy',SON([('$lt',.2)])),
                       ('$where' , '(this.config.image.sx / this.config.image.sy < 2.0) && (this.config.image.sx / this.config.image.sy > 1/2.0)')])),
      ('query',[SON([('image.object',SON([('$size',5)]))]),SON([('image.object',SON([('$size',7)]))]),SON([('image.object.type','rectangle')]),SON([('image.object.type','arc')])])
   ]),
   ]
   
   
}