import copy 

from bson import SON

from model_categories import MODEL_CATEGORIES as CAT

from config.helpers import uset, mixup
    
NTRAIN = 60
NTEST = 30
NUM_SPLITS = 10
OVERLAP = .75

tenway_cats = ['cars','planes','boats','guns','faces','chair',
               'table','reptiles','plants','cats_and_dogs']
               
               
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
for (ind1,cat1) in enumerate(tenway_cats):
    for cat2 in tenway_cats[ind1+1:]:
        ctask =  SON([('transform_average', SON([('transform_name','translation'),('max',True)])),
                     ('N',NUM_SPLITS), 
                     ('ntrain',NTRAIN),
                     ('ntest',NTEST),
                     ('overlap',OVERLAP),
                     ('universe',SON([('image.bg_id',SON([('$ne','gray.tdl')]))])),
                     ('query',[SON([('image.model_id',SON([('$in',CAT[cat1])]))]),
                               SON([('image.model_id',SON([('$in',CAT[cat2])]))])])
                ]) 
        mtask =  SON([('transform_average', SON([('transform_name','translation'),('max',True)])),
                     ('N',NUM_SPLITS), 
                     ('ntrain',NTRAIN),
                     ('ntest',NTEST),
                     ('overlap',OVERLAP),
                     ('universe',SON([('image.bg_id',SON([('$ne','gray.tdl')]))])),
                     ('query',[SON([('image.model_id',SON([('$in',k)]))]) for k in mixup(CAT,[[cat1],[cat2]])])
                ]) 
        
        t1 = copy.deepcopy(ctask)
        t1['universe'].update(trans_q)
        t1['task_label'] = cat1 + '/' + cat2 + ' trans'
        task_set.append(t1)
        
        t2 = copy.deepcopy(ctask)
        t2['universe'].update(inrot_q)
        t2['task_label'] = cat1 + '/' + cat2 + ' inrot'
        task_set.append(t2)
        
        t3 = copy.deepcopy(mtask)
        t3['universe'].update(trans_q)
        t3['task_label'] = cat1 + '/' + cat2 + ' trans mixed'
        task_set.append(t3)

         
    
config = {
'train_test' : [task_set]
}



