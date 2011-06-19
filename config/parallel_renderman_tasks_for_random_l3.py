from bson import SON

import model_categories as mc

config = {


'train_test' : [
   SON([
      ('N',3), 
      ('ntrain',320),
      ('ntest',160),
      ('universe',SON([('image.bg_id',SON([('$ne','gray.tdl')]))])),
      ('query',[SON([('image.model_id','face0001')]),
                SON([('image.model_id','face0002')]),
                SON([('image.model_id','face0003')]),
                SON([('image.model_id','face0004')]),
                SON([('image.model_id','face0005')]),
                SON([('image.model_id','face0006')]),
                SON([('image.model_id','face0007')]),
                SON([('image.model_id','face0008')]),
               ])
      ]),
   SON([
      ('N',3), 
      ('ntrain',400),
      ('ntest',200),
      ('universe',SON([('image.bg_id',SON([('$ne','gray.tdl')]))])),
      ('query',[SON([('image.model_id',SON([('$in',mc.MODEL_CATEGORIES['cars'])]))]),
                SON([('image.model_id',SON([('$in',mc.MODEL_CATEGORIES['planes'])]))]),
                SON([('image.model_id',SON([('$in',mc.MODEL_CATEGORIES['boats'])]))]),
                SON([('image.model_id',SON([('$in',mc.MODEL_CATEGORIES['guns'])]))]),
                SON([('image.model_id',SON([('$in',mc.MODEL_CATEGORIES['faces'])]))]),
                SON([('image.model_id',SON([('$in',mc.MODEL_CATEGORIES['chair'])]))]),
                SON([('image.model_id',SON([('$in',mc.MODEL_CATEGORIES['table'])]))]),
                SON([('image.model_id',SON([('$in',mc.MODEL_CATEGORIES['reptiles'])]))]),
                SON([('image.model_id',SON([('$in',mc.MODEL_CATEGORIES['plants'])]))]),
                SON([('image.model_id',SON([('$in',mc.MODEL_CATEGORIES['cats_and_dogs'])]))])
               ])
      ]),      
   ]
   
   
}


