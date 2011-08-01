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

task_cats = [['reptiles','planes']]

task_set = []
for cat1,cat2 in task_cats:
    
    base_task =  SON([('N',NUM_SPLITS), 
              ('ntrain',NTRAIN),
              ('ntest',NTEST),
              ('overlap',OVERLAP),
              ('classifier_kwargs',SON([('classifier_type','LRL'),('penalty','l2')])),        
              ('universe',SON([('image.bg_id','gray.tdl')])),
              ('query',[SON([('image.model_id',SON([('$in',CAT[cat1])]))]),
                       SON([('image.model_id',SON([('$in',CAT[cat2])]))])])
             ]) 
                
    
        
    t1 = copy.deepcopy(base_task)
    t1['universe'].update(trans_q)
    t1['task_label'] = cat1 + '/' + cat2 + ' trans'
    task_set.append(t1)

    t2 = copy.deepcopy(base_task)
    t2['universe'].update(inrot_q)
    t2['task_label'] = cat1 + '/' + cat2 + ' inrot'
    task_set.append(t2)
    

    
config = {
'train_test' : [task_set]
}



