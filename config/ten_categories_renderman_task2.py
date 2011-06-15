from bson import SON

import model_categories as mc

config = {


'train_test' : [

   SON([
      ('transform_average', SON([('transform_name','translation')])),
      ('N',1), 
      ('ntrain',240),
      ('ntest',42),
      ('universe',SON([('image.ty',SON([('$lt',.22),('$gt',-.22)])),
                       ('image.tz',SON([('$lt',.22),('$gt',-.22)]))])),
      ('query',[SON([('image.model_id',SON([('$in',mc.MODEL_CATEGORIES['cars'])]))]),
                SON([('image.model_id',SON([('$in',mc.MODEL_CATEGORIES['planes'])]))])
               ])
      ]),
   SON([
      ('N',1), 
      ('ntrain',240),
      ('ntest',42),
      ('universe',SON([('image.ty',SON([('$lt',.22),('$gt',-.22)])),
                       ('image.tz',SON([('$lt',.22),('$gt',-.22)]))])),
      ('query',[SON([('image.model_id',SON([('$in',mc.MODEL_CATEGORIES['cars'])]))]),
                SON([('image.model_id',SON([('$in',mc.MODEL_CATEGORIES['planes'])]))])
               ])
      ]),
   SON([
      ('N',1), 
      ('ntrain',240),
      ('ntest',42),
      ('universe',SON([('image.ty',SON([('$lt',.5),('$gt',-.5)])),
                       ('image.tz',SON([('$lt',.5),('$gt',-.5)]))])),
      ('query',[SON([('image.model_id',SON([('$in',mc.MODEL_CATEGORIES['cars'])]))]),
                SON([('image.model_id',SON([('$in',mc.MODEL_CATEGORIES['planes'])]))])
               ])
      ]),
   SON([
      ('N',2), 
      ('ntrain',100),
      ('ntest',30),
      ('universe',SON([('image.ty',SON([('$lt',.5),('$gt',-.5)])),
                       ('image.tz',SON([('$lt',.5),('$gt',-.5)])),
                       ('image.bg_id','gray.tdl')])),
      ('query',[SON([('image.model_id',SON([('$in',mc.MODEL_CATEGORIES['cars'])]))]),
                SON([('image.model_id',SON([('$in',mc.MODEL_CATEGORIES['planes'])]))])
               ])
      ]),
   SON([
      ('N',2), 
      ('ntrain',240),
      ('ntest',42),
      ('universe',SON([('image.ty',SON([('$lt',.22),('$gt',-.22)])),
                       ('image.tz',SON([('$lt',.22),('$gt',-.22)]))])),
      ('query',[SON([('image.model_id',SON([('$in',mc.MODEL_CATEGORIES['cars'])]))]),
                SON([('image.model_id',SON([('$in',mc.MODEL_CATEGORIES['planes'])])),
                SON([('image.model_id',SON([('$in',mc.MODEL_CATEGORIES['cats_and_dogs'])]))])
               ])
      ]),
      
   ]
   
   
}


