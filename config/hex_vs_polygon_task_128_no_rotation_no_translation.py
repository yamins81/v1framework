from bson import SON

config = {


'train_test' : [

   #sq vs polygon
   SON([
      ('transform_average', SON([('transform_name','translation')])),
      ('N',1), 
      ('ntrain',128),
      ('ntest',32),
      ('ntrain_pos',64),
      ('ntest_pos',16),
      ('universe',SON([('image.rxy',SON([('$lt',.2)])),
                       ('image.tx',SON([('$lt',.075),('$gt',-0.075)])),('image.ty',SON([('$lt',.075),('$gt',-.075)]))])),
      ('query',SON([('image.object',SON([('$size',8)]))]))
   ])
   
   ]
   
   
}