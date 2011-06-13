from bson import SON

config = {


'train_test' : [

   #sq vs polygon
   
   SON([
      ('transform_average', SON([('transform_name','translation_and_fourier')])),
      ('N',4), 
      ('ntrain',300),
      ('ntest',64),
      ('ntrain_pos',150),
      ('ntest_pos',32),
      ('universe',SON([('image.tx',SON([('$lt',.22),('$gt',-.22)])),('image.ty',SON([('$lt',.22),('$gt',-.22)]))])),
      ('query',SON([('image.object',SON([('$size',8)]))]))
   ]), 
   SON([
      ('transform_average', SON([('transform_name','translation_and_fourier')])),
      ('N',1), 
      ('ntrain',128),
      ('ntest',32),
      ('ntrain_pos',64),
      ('ntest_pos',16),
      ('universe',SON([('$or',[SON([('image.object',SON([('$size',8)]))]),SON([('image.object',SON([('$size',5)]))]),SON([('image.object.type','rectangle')]),SON([('image.object.type','arc')])]),
                       ('image.tx',SON([('$lt',.22),('$gt',-.22)])),
                       ('image.ty',SON([('$lt',.22),('$gt',-.22)]))])),
      ('query',SON([('image.object',SON([('$size',8)]))]))
   ]),  
   SON([
      ('transform_average', SON([('transform_name','translation_and_fourier')])),
      ('N',1), 
      ('ntrain',300),
      ('ntest',32),
      ('ntrain_pos',150),
      ('ntest_pos',16),
      ('universe',SON([('$or',[SON([('image.object',SON([('$size',8)]))]),SON([('image.object',SON([('$size',5)]))]),SON([('image.object.type','rectangle')])]),
                       ('image.tx',SON([('$lt',.22),('$gt',-.22)])),
                       ('image.ty',SON([('$lt',.22),('$gt',-.22)]))])),
      ('query',SON([('image.object',SON([('$size',8)]))]))
   ]),
   SON([
      ('transform_average', SON([('transform_name','translation_and_fourier')])),
      ('N',1), 
      ('ntrain',128),
      ('ntest',32),
      ('ntrain_pos',64),
      ('ntest_pos',16),
      ('universe',SON([('$or',[SON([('image.object',SON([('$size',8)]))]),SON([('image.object.type','arc')])]),
                       ('image.tx',SON([('$lt',.22),('$gt',-.22)])),
                       ('image.ty',SON([('$lt',.22),('$gt',-.22)]))])),
      ('query',SON([('image.object',SON([('$size',8)]))]))
   ]), 
   SON([
      ('transform_average', SON([('transform_name','translation_and_fourier')])),
      ('N',1), 
      ('ntrain',512),
      ('ntest',32),
      ('ntrain_pos',256),
      ('ntest_pos',16),
      ('universe',SON([('$or',[SON([('image.object',SON([('$size',8)]))]),SON([('image.object',SON([('$size',5)]))]),SON([('image.object.type','rectangle')]),SON([('image.object.type','arc')])]),
                       ('image.tx',SON([('$lt',.22),('$gt',-.22)])),
                       ('image.ty',SON([('$lt',.22),('$gt',-.22)]))])),
      ('query',SON([('image.object',SON([('$size',8)]))]))
   ]), 
   SON([
      ('transform_average', SON([('transform_name','translation_and_fourier')])),
      ('N',1), 
      ('ntrain',128),
      ('ntest',32),
      ('ntrain_pos',64),
      ('ntest_pos',16),
      ('universe',SON([('$or',[SON([('image.object',SON([('$size',8)]))]),SON([('image.object',SON([('$size',5)]))]),SON([('image.object.type','rectangle')]),SON([('image.object.type','arc')])]),
                       ('image.tx',SON([('$lt',.22),('$gt',-.22)])),
                       ('image.ty',SON([('$lt',.22),('$gt',-.22)])),
                       ('image.pattern.args',[1,0,0,1.0])])),
      ('query',SON([('image.object',SON([('$size',8)]))]))
   ]), 
   ]
   
   
}






