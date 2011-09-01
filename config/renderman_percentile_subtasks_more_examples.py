import copy 

from bson import SON

from model_categories import MODEL_CATEGORIES as CAT

from config.helpers import uset, mixup
    
NTRAIN = 100
NTEST = 30
NUM_SPLITS = 20               

import pymongo as pm
import numpy as np
import itertools

tenway_task =  SON([('N',NUM_SPLITS), 
		  ('ntrain',10*NTRAIN),
		  ('ntest',10*NTEST),
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
		 
reptile_plane_task =  SON([('N',NUM_SPLITS), 
		  ('ntrain',2*NTRAIN),
		  ('ntest',2*NTEST),
                  ('universe',SON([('image.bg_id','gray.tdl')])),                 
		  ('query',[SON([('image.model_id',SON([('$in',CAT['planes'])]))]),
                    SON([('image.model_id',SON([('$in',CAT['reptiles'])]))])
                   ])
		 ])
		 
base_tasks = [(tenway_task,'tenway'),(reptile_plane_task,'reptile/plane')]
			
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
              
invars = [(None,''),(trans_q,'trans'),(inrot_q,'inrot')]

layers = [(None,''),([-1,0,1],'upto layer 1'),([-1,0,1,2],'upto layer 2')]
pcts = [(None,''),([1],'pct 1'),([2],'pct 2'),([3],'pct 3'),([2,3],'pct 2 & 3')]
subranges = [(SON([('layers',l[0]),('percts',p[0])]),(l[1] + ' ' + p[1]).strip()) for l in layers for p in pcts]
            
prod = itertools.product(base_tasks,invars,subranges)
	
task_set = []
for (t,inv,sr) in prod:
    task = copy.deepcopy(t[0])
    if inv[0]:
        task['universe'].update(inv[0])
    if sr[0]:
        task['feature_postprocess'] = SON([('transform_name','subranges')])
        task['feature_postprocess'].update(sr[0])
        
    task['task_label'] = (t[1] + ' ' + inv[1] + ' ' + sr[1]).strip().replace('  ',' ')
    
    task_set.append(task)

config = {
'train_test' : task_set
}



