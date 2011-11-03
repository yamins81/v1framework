import copy 

from bson import SON

import model_categories as mc
      
TYPES = [u'',
         u'Boat',
         u'Car',
         u'Container',
         u'Cyclist',
         u'Person',
         u'Plane',
         u'Truck']

EMPIRICAL_COUNTS = [8718, 3, 4400, 47, 31, 135, 477, 18]
TOTAL = float(sum(EMPIRICAL_COUNTS))
BALANCE = [c/TOTAL for c in EMPIRICAL_COUNTS[:-1]]

task = SON([
      ('N',10), 
      ('ntrain', 5000),
      ('ntest', 500),
      ('balance',BALANCE),
      ('universe',SON([('image.correctness',SON([('$ne',0)]))])),
      ('query',[SON([('image.ObjectType',t)]) for t in TYPES])
      ])

CP_TYPES = [u'',u'Car',u'Plane',]
CP_EMPIRICAL_COUNTS = [8718, 4400, 477]
CP_TOTAL = float(sum(CP_EMPIRICAL_COUNTS))
CP_BALANCE = [c/CP_TOTAL for c in CP_EMPIRICAL_COUNTS[:-1]]

car_plane_task = SON([
      ('N',10), 
      ('ntrain', 10000),
      ('ntest', 500),
      ('balance',CP_BALANCE),
      ('universe',SON([('image.correctness',SON([('$ne',0)]))])),
      ('query',[SON([('image.ObjectType',t)]) for t in CP_TYPES])
      ])

cp_task_fewer = copy.deepcopy(car_plane_task)
cp_task_fewer['ntrain'] = 1000

cp_task_medium = copy.deepcopy(car_plane_task)
cp_task_medium['ntrain'] = 5000
cp_task_medium2 = copy.deepcopy(car_plane_task)
cp_task_medium2['ntrain'] = 5000
cp_task_medium2['ntest'] = 1000

CPP_TYPES = [u'',u'Car',u'Person',u'Plane']
CPP_EMPIRICAL_COUNTS = [8718, 4400, 135, 477]
CPP_TOTAL = float(sum(CPP_EMPIRICAL_COUNTS))
CPP_BALANCE = [c/CPP_TOTAL for c in CPP_EMPIRICAL_COUNTS[:-1]]

car_plane_person_task = SON([
          ('N',10),
                ('ntrain', 5000),
                ('ntest', 1000),
                ('balance',CPP_BALANCE),
                ('universe',SON([('image.correctness',SON([('$ne',0)]))])),
                ('query',[SON([('image.ObjectType',t)]) for t in CPP_TYPES])
                ])

BINARY_TYPES = [1,-1]
BINARY_EMPIRICAL_COUNTS = [8718,5111]
BINARY_TOTAL = float(sum(BINARY_EMPIRICAL_COUNTS))
BINARY_BALANCE = [c/BINARY_TOTAL for c in BINARY_EMPIRICAL_COUNTS[:-1]]
binary_task = SON([ ('N',10),
                              ('ntrain', 2500),
                              ('ntest', 500),
                              ('balance',BINARY_BALANCE),
                              ('universe',SON([('image.correctness',SON([('$ne',0)]))])),
                              ('query',[SON([('image.correctness',t)]) for t in BINARY_TYPES])
                              ])

binary_5fold_task = SON([('kfold',5),
                         ('universe',SON([('image.correctness',SON([('$ne',0)]))])),
                         ('query',[SON([('image.correctness',t)]) for t in BINARY_TYPES])
                         ])


config = {
'train_test' : [task, car_plane_task, cp_task_fewer, cp_task_medium, cp_task_medium2, car_plane_person_task, binary_task, binary_5fold_task]
   
}



