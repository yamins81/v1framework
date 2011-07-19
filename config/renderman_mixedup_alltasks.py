import copy 

from bson import SON

from model_categories import MODEL_CATEGORIES as CAT

from config.helpers import uset, mixup
    

tenway_cats = [['cars'],['planes'],['boats'],['guns'],['faces'],['chair'],
                    ['table'],['reptiles'],['plants'],['cats_and_dogs']]
    
tenway_task = SON([
      ('transform_average', SON([('transform_name','translation'),('max',True)])),
      ('N',10), 
      ('ntrain',300),
      ('ntest',150),
      ('overlap',.75),
      ('universe',SON([('image.bg_id','gray.tdl')])),
      ('query',[SON([('image.model_id',SON([('$in',uset(CAT,k))]))]) for k in tenway_cats])
      ])      

tenway_mixup_task = SON([
      ('transform_average', SON([('transform_name','translation'),('max',True)])),
      ('N',10), 
      ('ntrain',300),
      ('ntest',150),
      ('overlap',.75),
      ('universe',SON([('image.bg_id','gray.tdl')])),
      ('query',[SON([('image.model_id',SON([('$in',k)]))]) for k in mixup(CAT,tenway_cats)])
      ])      

      
car_plane_cats = [['cars'],['planes']]

car_vs_plane_task = SON([
      ('transform_average', SON([('transform_name','translation'),('max',True)])),
      ('N',10), 
      ('ntrain',60),
      ('ntest',30),
      ('overlap',.75),
      ('universe',SON([('image.bg_id','gray.tdl')])),
      ('query',[SON([('image.model_id',SON([('$in',uset(CAT,k))]))]) for k in car_plane_cats])
      ])  
      
car_vs_plane_mixup_task = SON([
      ('transform_average', SON([('transform_name','translation'),('max',True)])),
      ('N',10), 
      ('ntrain',60),
      ('ntest',30),
      ('overlap',.75),
      ('universe',SON([('image.bg_id','gray.tdl')])),
      ('query',[SON([('image.model_id',SON([('$in',k)]))]) for k in mixup(CAT,car_plane_cats)])
      ])  
      
fav_cats = [['chair','table'],['cats_and_dogs','reptiles'],['cars','boats']]

furniture_vs_animals_vs_vehicles_task =  SON([
      ('transform_average', SON([('transform_name','translation'),('max',True)])),
      ('N',10), 
      ('ntrain',90),
      ('ntest',45),
      ('overlap',.75),
      ('universe',SON([('image.bg_id','gray.tdl')])),
      ('query',[SON([('image.model_id',SON([('$in',uset(CAT,k))]))]) for k in fav_cats])
      ])  

furniture_vs_animals_vs_vehicles_mixup_task =  SON([
      ('transform_average', SON([('transform_name','translation'),('max',True)])),
      ('N',10), 
      ('ntrain',90),
      ('ntest',45),
      ('overlap',.75),
      ('universe',SON([('image.bg_id','gray.tdl')])),
      ('query',[SON([('image.model_id',SON([('$in',k)]))]) for k in mixup(CAT,fav_cats)])
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
      
basetask_set = [tenway_task,tenway_mixup_task,car_vs_plane_task,
                car_vs_plane_mixup_task, furniture_vs_animals_vs_vehicles_task,
                furniture_vs_animals_vs_vehicles_mixup_task]
                

task_set = []
for t in basetask_set:
    task_set.append(t)
    t1 = copy.deepcopy(t)
    t1['universe'].update(trans_q)
    task_set.append(t1)
    t2 = copy.deepcopy(t)
    t2['universe'].update(inrot_q)
    task_set.append(t2)

config = {
'train_test' : task_set
}



