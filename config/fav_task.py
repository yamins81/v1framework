import copy 

from bson import SON

from model_categories import MODEL_CATEGORIES as CAT
      
fav_task = SON([
      ('transform_average', SON([('transform_name','translation'),('max',True)])),
      ('N',10), 
      ('ntrain',300),
      ('ntest',150),
      ('overlap',.75),
      ('universe',SON([('image.bg_id','gray.tdl')])),
      ('query',[SON([('image.model_id',SON([('$in',CAT['cars']+CAT['boats'])]))]),
                SON([('image.model_id',SON([('$in',CAT['cats_and_dogs']+CAT['reptiles'])]))]),
                SON([('image.model_id',SON([('$in',CAT['table']+CAT['chair'])]))]),
               ])
      ])      
        

config = {

'train_test' : [fav_task
               ]
   
}



