from bson import SON

config = {


'train_test' : [

   #sq vs polygon
   SON([
      ('N',5), 
      ('transform_average', SON([('transform_name','translation')])),
      ('ntrain',64),
      ('ntest',32),
      ('ntrain_pos',32),
      ('ntest_pos',16),
      ('universe',SON([('image.tx',SON([('$lt',.22),('$gt',-.22)])),('image.ty',SON([('$lt',.22),('$gt',-.22)]))])),
      ('query',SON([('image.object.type','rectangle')]))
   ])
   
   ]
   
   
}