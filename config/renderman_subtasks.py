import copy 

from bson import SON

import model_categories as mc
      
tenway_task = SON([
      ('transform_average', SON([('transform_name','translation'),('max',True)])),
      ('N',3), 
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

outrot_q = SON([('image.ty',SON([('$exists',False)])),
               ('image.tz',SON([('$exists',False)])),
               ('image.tx',SON([('$exists',False)])),
               ('image.s',SON([('$exists',False)])),
               ('image.rxy',SON([('$exists',True)])),
               ('image.rxz',SON([('$exists',True)])),
               ('image.ryz',SON([('$exists',False)])),
              ])

scale_q = SON([('image.ty',SON([('$exists',False)])),
               ('image.tz',SON([('$exists',False)])),
               ('image.tx',SON([('$exists',False)])),
               ('image.s',SON([('$exists',True)])),
               ('image.rxy',SON([('$exists',False)])),
               ('image.rxz',SON([('$exists',False)])),
               ('image.ryz',SON([('$exists',False)])),
              ])    

scale_inrot_q = SON([('image.ty',SON([('$exists',False)])),
               ('image.tz',SON([('$exists',False)])),
               ('image.tx',SON([('$exists',False)])),
               ('image.s',SON([('$exists',True)])),
               ('image.rxy',SON([('$exists',False)])),
               ('image.rxz',SON([('$exists',False)])),
               ('image.ryz',SON([('$exists',True)])),
              ])    

trans_inrot_q = SON([('image.ty',SON([('$exists',True)])),
               ('image.tz',SON([('$exists',True)])),
               ('image.tx',SON([('$exists',False)])),
               ('image.s',SON([('$exists',False)])),
               ('image.rxy',SON([('$exists',False)])),
               ('image.rxz',SON([('$exists',False)])),
               ('image.ryz',SON([('$exists',True)])),
              ])

allrot_q = SON([('image.ty',SON([('$exists',False)])),
               ('image.tz',SON([('$exists',False)])),
               ('image.tx',SON([('$exists',False)])),
               ('image.s',SON([('$exists',False)])),
               ('image.rxy',SON([('$exists',True)])),
               ('image.rxz',SON([('$exists',True)])),
               ('image.ryz',SON([('$exists',True)])),
              ])
    

t1 = copy.deepcopy(tenway_task)
t1['universe'].update(trans_q)
t2 = copy.deepcopy(tenway_task)
t2['universe'].update(inrot_q)
t3 = copy.deepcopy(tenway_task)
t3['universe'].update(outrot_q)
t4 = copy.deepcopy(tenway_task)
t4['universe'].update(scale_q)
t5 = copy.deepcopy(tenway_task)
t5['universe'].update(scale_inrot_q)
t6 = copy.deepcopy(tenway_task)
t6['universe'].update(trans_inrot_q)
t7 = copy.deepcopy(tenway_task)
t7['universe'].update(allrot_q)
    
config = {

'train_test' : [t1,t2,t3,t4,t5,t6,t7]  
   
}



