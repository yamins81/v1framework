from bson import SON

config = {


'train_test' : [

   #sq vs polygon
   
   SON([
      ('transform_average', SON([('transform_name','translation_and_fourier')])),
      ('N',1), 
      ('ntrain',300),
      ('ntest',40),
      ('ntrain_pos',150),
      ('ntest_pos',20),
      ('universe',SON([('image.tx',SON([('$lt',.22),('$gt',-.22)])),('image.ty',SON([('$lt',.22),('$gt',-.22)]))])),
      ('query',SON([('image.object',SON([('$size',8)]))]))
   ]), 
   SON([
      ('transform_average', SON([('transform_name','translation_and_fourier')])),
      ('N',1), 
      ('ntrain',300),
      ('ntest',40),
      ('ntrain_pos',150),
      ('ntest_pos',20),
      ('universe',SON([('image.tx',SON([('$lt',.22),('$gt',-.22)])),('image.ty',SON([('$lt',.22),('$gt',-.22)]))])),
      ('query',SON([('image.object.type','rectangle')]))
   ]), 
    SON([
      ('transform_average', SON([('transform_name','translation_and_fourier')])),
      ('N',1), 
      ('ntrain',300),
      ('ntest',40),
      ('ntrain_pos',150),
      ('ntest_pos',20),
      ('universe',SON([('image.tx',SON([('$lt',.22),('$gt',-.22)])),('image.ty',SON([('$lt',.22),('$gt',-.22)]))])),
      ('query',SON([('image.object',SON([('$size',5)]))]))
   ]),   

   ]
   
   
}