import copy 

from bson import SON

from model_categories import MODEL_CATEGORIES as CAT

from config.helpers import uset, mixup
    
NTRAIN = 60
NTEST = 30
NUM_SPLITS = 10
OVERLAP = .75



               
trans_q = SON([('image.ty',SON([('$exists',True)])),
               ('image.tz',SON([('$exists',True)])),
               ('image.tx',SON([('$exists',False)])),
               ('image.s',SON([('$exists',False)])),
               ('image.rxy',SON([('$exists',False)])),
               ('image.rxz',SON([('$exists',False)])),
               ('image.ryz',SON([('$exists',False)])),
              ])

inrot_q = SON([('image.ty',SON([('$exists',False)])),
               ('image.tz',SON([('$exists',False)])),
               ('image.tx',SON([('$exists',False)])),
               ('image.s',SON([('$exists',False)])),
               ('image.rxy',SON([('$exists',False)])),
               ('image.rxz',SON([('$exists',False)])),
               ('image.ryz',SON([('$exists',True)])),
              ])


task_set = []

base_task =  SON([('N',NUM_SPLITS), 
		  ('ntrain',NTRAIN),
		  ('ntest',NTEST),
		  ('overlap',OVERLAP),
		  ('universe',SON([('image.bg_id','gray.tdl')])),
		  ('query',[SON([('image.model_id','MB29694')]),
				   SON([('image.model_id',SON([('$in',CAT['planes'])]))])])
		 ]) 
			
	
t1 = copy.deepcopy(base_task)
t1['universe'].update(trans_q)
t1['task_label'] = 'MB29694/planes trans'
task_set.append(t1)

t2 = copy.deepcopy(base_task)
t2['universe'].update(inrot_q)
t2['task_label'] = 'MB29694/planes inrot'
task_set.append(t2)


    
config = {
'train_test' : [task_set]
}



