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
                  ('universe',SON([('image.bg_id','gray.tdl')])),                 
		  ('query',[SON([('image.model_id',SON([('$in',CAT['cars'])]))]),
                    SON([('image.model_id',SON([('$in',CAT['planes'])]))]),
                    SON([('image.model_id',SON([('$in',CAT['boats'])]))]),
                    SON([('image.model_id',SON([('$in',CAT['guns'])]))]),
                    SON([('image.model_id',SON([('$in',CAT['faces'])]))]),
                    SON([('image.model_id',SON([('$in',CAT['chair'])]))]),
                    SON([('image.model_id',SON([('$in',CAT['table'])]))]),
                    SON([('image.model_id',SON([('$in',CAT['reptiles'])]))]),
                    SON([('image.model_id',SON([('$in',CAT['plants'])]))]),
                    SON([('image.model_id',SON([('$in',CAT['cats_and_dogs'])]))])
                   ])
		 ]) 
			
	
config = {
'train_test' : [tenway_task]
}



