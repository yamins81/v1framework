import copy 

from bson import SON

from model_categories import MODEL_CATEGORIES as CAT

from config.helpers import uset, mixup
    
NTRAIN = 300
NTEST = 150
NUM_SPLITS = 20               


tenway_task =  SON([('N',NUM_SPLITS), 
		  ('ntrain',NTRAIN),
		  ('ntest',NTEST),
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
			
	
config = {
'train_test' : [tenway_task]
}



