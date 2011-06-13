from bson import SON

config = {


'train_test' : [

   #sq vs polygon
   SON([
      ('transform_average', SON([('transform_name','translation')])),
      ('N',1), 
      ('ntrain',64),
      ('ntest',32),
      ('ntrain_pos',32),
      ('ntest_pos',16),
      ('universe',SON([('image.rxy',SON([('$lt',.2)])),
                       ('image.tx',SON([('$lt',.22),('$gt',-.22)])),('image.ty',SON([('$lt',.22),('$gt',-.22)]))])),
      ('query',SON([('image.object',SON([('$size',8)]))]))
   ]),
   SON([
      ('transform_average', SON([('transform_name','translation')])),
      ('N',1), 
      ('ntrain',256),
      ('ntest',32),
      ('ntrain_pos',128),
      ('ntest_pos',16),
      ('universe',SON([('image.rxy',SON([('$lt',.2)])),
                       ('image.tx',SON([('$lt',.22),('$gt',-.22)])),('image.ty',SON([('$lt',.22),('$gt',-.22)]))])),
      ('query',SON([('image.object',SON([('$size',8)]))]))
   ]),
   SON([
      ('transform_average', SON([('transform_name','translation')])),
      ('N',1),
      ('ntrain',128),
      ('ntest',32),
      ('ntrain_pos',64),
      ('ntest_pos',16),
      ('universe',SON([('image.rxy',SON([('$lt',.2)])),
                       ('image.pattern.args',[1,0,0,1.0]),
                       ('image.tx',SON([('$lt',.22),('$gt',-.22)])),('image.ty',SON([('$lt',.22),('$gt',-.22)]))])),
      ('query',SON([('image.object',SON([('$size',8)]))]))
   ]),
   SON([
      ('transform_average', SON([('transform_name','translation')])),
      ('N',1), 
      ('ntrain',128),
      ('ntest',32),
      ('ntrain_pos',64),
      ('ntest_pos',16),
      ('universe',SON([('image.rxy',SON([('$lt',.1)])),
                       ('image.tx',SON([('$lt',.22),('$gt',-.22)])),('image.ty',SON([('$lt',.22),('$gt',-.22)]))])),
      ('query',SON([('image.object',SON([('$size',8)]))]))
   ]),
   SON([
      ('transform_average', SON([('transform_name','translation_and_fourier')])),
      ('N',1), 
      ('ntrain',128),
      ('ntest',32),
      ('ntrain_pos',64),
      ('ntest_pos',16),
      ('universe',SON([('image.rxy',SON([('$lt',1.57)])),
                       ('image.tx',SON([('$lt',.22),('$gt',-.22)])),('image.ty',SON([('$lt',.22),('$gt',-.22)]))])),
      ('query',SON([('image.object',SON([('$size',8)]))]))
   ]),   
   ]
   
   
}