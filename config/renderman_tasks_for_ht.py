import copy 

from bson import SON

import model_categories as mc
      
tenway_task = SON([
      ('transform_average', SON([('transform_name','translation'),('max',True)])),
      ('N',5), 
      ('ntrain',400),
      ('ntest',200),
      ('universe',SON([('image.bg_id','gray.tdl')])),
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
      ])      
        
car_plane_task = SON([
      ('transform_average', SON([('transform_name','translation'),('max',True)])),
      ('N',5), 
      ('ntrain',80),
      ('ntest',40),
      ('universe',SON([('image.bg_id','gray.tdl')])),
      ('query',[SON([('image.model_id',SON([('$in',mc.MODEL_CATEGORIES['cars'])]))]),
                SON([('image.model_id',SON([('$in',mc.MODEL_CATEGORIES['planes'])]))]),
               ])
      ])   

config = {

'train_test' : [tenway_task,
                car_plane_task
               ]
   
}



