from bson import SON

import copy

fiveway_task =   SON([
      ('N',3), 
      ('transform_average', SON([('transform_name','translation'),('max',True)])), 
      ('ntrain',200),
      ('ntest',100),
      ('universe',SON([('image.tx',SON([('$lt',.22),('$gt',-.22)])),
                       ('image.ty',SON([('$lt',.22),('$gt',-.22)])),
                       ('$where' , '(this.config.image.sx / this.config.image.sy < 2.0) && (this.config.image.sx / this.config.image.sy > 1/2.0)')])),
      ('query',[SON([('image.object',SON([('$size',8)]))]),SON([('image.object',SON([('$size',5)]))]),SON([('image.object',SON([('$size',7)]))]),SON([('image.object.type','rectangle')]),SON([('image.object.type','arc')])])
   ])

rect_task =   SON([
      ('transform_average', SON([('transform_name','translation'),('max',True)])),
      ('N',3), 
      ('ntrain',80),
      ('ntest',40),
      ('universe',SON([('image.tx',SON([('$lt',.22),('$gt',-.22)])),
                       ('image.ty',SON([('$lt',.22),('$gt',-.22)])),
                       ('$where' , '(this.config.image.sx / this.config.image.sy < 2.0) && (this.config.image.sx / this.config.image.sy > 1/2.0)')])),
      ('query',SON([('image.object.type','rectangle')]))
   ])


trans_q = SON([('image.sx',SON([('$lt',1.2),('$gt',1/1.2)])),
               ('image.sy',SON([('$lt',1.2),('$gt',1/1.2)])),
               ('image.rxy',SON([('$lt',.2)])),
               ])
               
trans_q_1 = SON([('image.sx',SON([('$lt',1.25),('$gt',1/1.25)])),
               ('image.sy',SON([('$lt',1.25),('$gt',1/1.25)])),
               ('image.rxy',SON([('$lt',.2)])),
               ])               
               
rot_q = SON([('image.tx',SON([('$lt',.05),('$gt',-.05)])),
               ('image.ty',SON([('$lt',.05),('$gt',-.05)])),
               ('image.sx',SON([('$lt',1.2),('$gt',1/1.2)])),
               ('image.sy',SON([('$lt',1.2),('$gt',1/1.2)])),
               ])
               
scale_q = SON([('image.tx',SON([('$lt',.06),('$gt',-.06)])),
               ('image.ty',SON([('$lt',.06),('$gt',-.06)])),
               ('image.rxy',SON([('$lt',.225)])),
               
scale_q_1 = SON([('image.tx',SON([('$lt',.08),('$gt',-.08)])),
               ('image.ty',SON([('$lt',.08),('$gt',-.08)])),
               ('image.rxy',SON([('$lt',.2)])),
               ])

scale_rot_q = SON([('image.tx',SON([('$lt',.015),('$gt',-.015)])),
                   ('image.ty',SON([('$lt',.015),('$gt',-.015)])),
               ])

trans_rot_q = SON([('image.sx',SON([('$lt',1.05),('$gt',1/1.05)])),
                   ('image.sy',SON([('$lt',1.05),('$gt',1/1.05)])),
               ])
 
t1 = copy.deepcopy(fiveway_task)
t1['universe'].update(trans_q_1)
t2 = copy.deepcopy(fiveway_task)
t2['universe'].update(rot_q)
t3 = copy.deepcopy(fiveway_task)
t3['universe'].update(scale_q_1)
t4 = copy.deepcopy(fiveway_task)
t4['universe'].update(scale_rot_q)
t5 = copy.deepcopy(fiveway_task)
t5['universe'].update(trans_rot_q)

t6 = copy.deepcopy(rect_task)
t6['universe'].update(trans_q)
t7 = copy.deepcopy(rect_task)
t7['universe'].update(rot_q)
t8 = copy.deepcopy(rect_task)
t8['universe'].update(scale_q)
t9 = copy.deepcopy(rect_task)
t9['universe'].update(scale_rot_q)
t10 = copy.deepcopy(rect_task)
t10['universe'].update(trans_rot_q)



config = {

'train_test' : [t1,t2,t3,t4,t5,
                t6,t7,t8,t9,t10]
                
}