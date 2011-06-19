from bson import SON

import model_categories as mc

config = {


'train_test' : [

   SON([
      ('N',3), 
      ('ntrain',400),
      ('ntest',80),
      ('universe',SON([('image.ty',SON([('$lt',.6),('$gt',-.6)])),
                       ('image.tz',SON([('$lt',.6),('$gt',-.6)]))])),
      ('query',[SON([('image.model_id',SON([('$in',mc.MODEL_CATEGORIES['cars'])]))]),
                SON([('image.model_id',SON([('$in',mc.MODEL_CATEGORIES['planes'])]))]),
                SON([('image.model_id',SON([('$in',mc.MODEL_CATEGORIES['boats'])]))]),
                SON([('image.model_id',SON([('$in',mc.MODEL_CATEGORIES['guns'])]))]),
                SON([('image.model_id',SON([('$in',mc.MODEL_CATEGORIES['faces'])]))]),
                SON([('image.model_id',SON([('$in',mc.MODEL_CATEGORIES['chair'])]))]),
                SON([('image.model_id',SON([('$in',mc.MODEL_CATEGORIES['table'])]))]),
                SON([('image.model_id',SON([('$in',mc.MODEL_CATEGORIES['cats_and_dogs'])]))])
               ])
      ]),
      
   ]
   
   
}


