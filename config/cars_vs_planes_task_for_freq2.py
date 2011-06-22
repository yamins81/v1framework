from bson import SON

import model_categories as mc

config = {


'train_test' : [
        
   SON([
      ('N',1), 
      ('ntrain',80),
      ('ntest',40),
      ('universe',SON([('image.bg_id',SON([('$ne','gray.tdl')]))])),
      ('query',[SON([('image.model_id',SON([('$in',mc.MODEL_CATEGORIES['cars'])]))]),
                SON([('image.model_id',SON([('$in',mc.MODEL_CATEGORIES['planes'])]))])
               ])
      ]),      
   ]
   
   
}