import copy 

from bson import SON

import model_categories as mc
      
eightface_task = SON([
      ('transform_average', SON([('transform_name','translation'),('max',True)])),
      ('N',3), 
      ('ntrain',320),
      ('ntest',160),
      ('universe',SON([('image.bg_id','gray.tdl')])),
      ('query',[SON([('image.model_id','face1')]),
                SON([('image.model_id','face2')]),
                SON([('image.model_id','face3')]),
                SON([('image.model_id','face4')]),
                SON([('image.model_id','face5')]),
                SON([('image.model_id','face6')]),
                SON([('image.model_id','face7')]),
                SON([('image.model_id','face8')]),
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
    

t0 = eightface_task
t1 = copy.deepcopy(eightface_task)
t1['universe'].update(trans_q)
t2 = copy.deepcopy(eightface_task)
t2['universe'].update(inrot_q)
t3 = copy.deepcopy(eightface_task)
t3['universe'].update(outrot_q)
t4 = copy.deepcopy(eightface_task)
t4['universe'].update(scale_q)
t5 = copy.deepcopy(eightface_task)
t5['universe'].update(scale_inrot_q)
t6 = copy.deepcopy(eightface_task)
t6['universe'].update(trans_inrot_q)
t7 = copy.deepcopy(eightface_task)
t7['universe'].update(allrot_q)



config = {

'train_test' : [t0,t1,t2,t3,t4,t5,t6,t7]
   
}



