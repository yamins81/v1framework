from bson import SON

config = {


'train_test' : [

   #sq vs polygon
   SON([
      ('transform_average', SON([('transform_name','translation_and_fourier')])),
      ('N',2), 
      ('ntrain',64),
      ('ntest',32),
      ('ntrain_pos',32),
      ('ntest_pos',16),
      ('universe',SON([('image.tx',SON([('$lt',.22),('$gt',-.22)])),('image.ty',SON([('$lt',.22),('$gt',-.22)]))])),
      ('query',SON([('image.object',SON([('$size',8)]))]))
   ]),
    SON([
      ('transform_average', SON([('transform_name','translation_and_fourier')])),
      ('N',1), 
      ('ntrain',256),
      ('ntest',32),
      ('ntrain_pos',128),
      ('ntest_pos',16),
      ('universe',SON([('image.tx',SON([('$lt',.22),('$gt',-.22)])),('image.ty',SON([('$lt',.22),('$gt',-.22)]))])),
      ('query',SON([('image.object',SON([('$size',8)]))]))
   ]) ,
    SON([
      ('transform_average', SON([('transform_name','translation_and_fourier')])),
      ('N',1), 
      ('ntrain',512),
      ('ntest',32),
      ('ntrain_pos',256),
      ('ntest_pos',16),
      ('universe',SON([('image.tx',SON([('$lt',.22),('$gt',-.22)])),('image.ty',SON([('$lt',.22),('$gt',-.22)]))])),
      ('query',SON([('image.object',SON([('$size',8)]))]))
   ]) 
   
   ]
   
   
}