from bson import SON

import model_categories as mc

config = {


'train_test' : [
   
   SON([
      ('transform_average', SON([('transform_name','translation'),('max',True)])),
      ('N',2), 
      ('ntrain',80),
      ('ntest',40),
      ('universe',SON([('image.bg_id',SON([('$ne','gray.tdl')]))])),
      ('query',[SON([('image.model_id',SON([('$in',mc.MODEL_CATEGORIES['cars'])]))]),
                SON([('image.model_id',SON([('$in',mc.MODEL_CATEGORIES['planes'])]))])
               ])
      ]),      
   
   ]
   
   
}