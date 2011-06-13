from bson import SON

config = {


'train_test' : [

   #sq vs polygon
   SON([
      ('N',10), 
      ('ntrain',128),
      ('ntest',32),
      ('ntrain_pos',64),
      ('universe',SON([('image.rxy',SON([('$lt',.1)])),
                       ('image.tx',SON([('$lt',.22),('$gt',-.22)])),
                       ('image.ty',SON([('$lt',.22),('$gt',-.22)]))
                      ])),
      ('query',SON([('image.object.type','rectangle')]))
   ]),
   
   ]
   
   
}