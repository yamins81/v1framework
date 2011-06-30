import copy 

from bson import SON

import model_categories as mc
      
tenway_task = SON([
      ('transform_average', SON([('transform_name','translation'),('max',True)])),
      ('N',3), 
      ('ntrain',400),
      ('ntest',200),
      ('universe',SON([('image.bg_id',SON([('$ne','gray.tdl')]))])),
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
      
car_vs_plane_task =  SON([
      ('transform_average', SON([('transform_name','translation'),('max',True)])),
      ('N',3), 
      ('ntrain',80),
      ('ntest',40),
      ('universe',SON([('image.bg_id','gray.tdl')])),
      ('query',[SON([('image.model_id',SON([('$in',mc.MODEL_CATEGORIES['cars'])]))]),
                SON([('image.model_id',SON([('$in',mc.MODEL_CATEGORIES['planes'])]))])
               ])
      ])      
      
chair_vs_table_task =  SON([
      ('transform_average', SON([('transform_name','translation'),('max',True)])),
      ('N',3), 
      ('ntrain',80),
      ('ntest',40),
      ('universe',SON([('image.bg_id','gray.tdl')])),
      ('query',[SON([('image.model_id',SON([('$in',mc.MODEL_CATEGORIES['chair'])]))]),
                SON([('image.model_id',SON([('$in',mc.MODEL_CATEGORIES['table'])]))])
               ])
      ])      

furniture_vs_animals_vs_vehicles_task =  SON([
      ('transform_average', SON([('transform_name','translation'),('max',True)])),
      ('N',3), 
      ('ntrain',80),
      ('ntest',40),
      ('universe',SON([('image.bg_id','gray.tdl')])),
      ('query',[SON([('image.model_id',SON([('$in',mc.MODEL_CATEGORIES['chair'] + mc.MODEL_CATEGORIES['table'])]))]),
                SON([('image.model_id',SON([('$in',mc.MODEL_CATEGORIES['cats_and_dogs'] + mc.MODEL_CATEGORIES['reptiles'])]))]),
                SON([('image.model_id',SON([('$in',mc.MODEL_CATEGORIES['cars'] + mc.MODEL_CATEGORIES['boats'] + mc.MODEL_CATEGORIES['planes'])]))])
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

t8 = copy.deepcopy(car_vs_plane_task)
t8['universe'].update(trans_q)
t9 = copy.deepcopy(car_vs_plane_task)
t9['universe'].update(inrot_q)
t10 = copy.deepcopy(car_vs_plane_task)
t10['universe'].update(outrot_q)
t11 = copy.deepcopy(car_vs_plane_task)
t11['universe'].update(scale_q)
t12 = copy.deepcopy(car_vs_plane_task)
t12['universe'].update(scale_inrot_q)
t13 = copy.deepcopy(car_vs_plane_task)
t13['universe'].update(trans_inrot_q)
t14 = copy.deepcopy(car_vs_plane_task)
t14['universe'].update(allrot_q)

t15 = copy.deepcopy(chair_vs_table_task)
t15['universe'].update(trans_q)
t16 = copy.deepcopy(chair_vs_table_task)
t16['universe'].update(inrot_q)
t17 = copy.deepcopy(chair_vs_table_task)
t17['universe'].update(outrot_q)
t18 = copy.deepcopy(chair_vs_table_task)
t18['universe'].update(scale_q)
t19 = copy.deepcopy(chair_vs_table_task)
t19['universe'].update(scale_inrot_q)
t20 = copy.deepcopy(chair_vs_table_task)
t20['universe'].update(trans_inrot_q)
t21 = copy.deepcopy(chair_vs_table_task)
t21['universe'].update(allrot_q)

t22 = copy.deepcopy(furniture_vs_animals_vs_vehicles_task)
t22['universe'].update(trans_q)
t23 = copy.deepcopy(furniture_vs_animals_vs_vehicles_task)
t23['universe'].update(inrot_q)
t24 = copy.deepcopy(furniture_vs_animals_vs_vehicles_task)
t24['universe'].update(outrot_q)
t25 = copy.deepcopy(furniture_vs_animals_vs_vehicles_task)
t25['universe'].update(scale_q)
t26 = copy.deepcopy(furniture_vs_animals_vs_vehicles_task)
t26['universe'].update(scale_inrot_q)
t27 = copy.deepcopy(furniture_vs_animals_vs_vehicles_task)
t27['universe'].update(trans_inrot_q)
t28 = copy.deepcopy(furniture_vs_animals_vs_vehicles_task)
t28['universe'].update(allrot_q)


config = {

'train_test' : [t1,t2,t3,t4,t5,t6,t7,
                t8,t9,t10,t11,t12,t13,t14,
                t15,t17,t17,t18,t19,t20,t21,
                t22,t23,t24,t25,t26,t27,t28]
   
}



