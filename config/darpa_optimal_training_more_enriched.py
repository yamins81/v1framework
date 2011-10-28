import copy 

from bson import SON

import model_categories as mc
      
TYPES = [u'Boat',
         u'Car',
         u'Container',
         u'Cyclist',
         u'Helicopter',
         u'Person',
         u'Plane',
         u'Tractor-Trailer',
         u'Truck',
         []]

BALANCE = [0.0005,
            0.15,
            0.0025,
            0.001,
            0.0001,
            0.001,
            0.01,
            0.0001,
            0.0005]

BALANCE2 = [0.0005,
            0.20,
            0.005,
            0.001,
            0.0001,
            0.001,
            0.03,
            0.0001,
            0.0005]

BALANCE3 = [0.0005,
            0.25,
            0.01,
            0.001,
            0.0001,
            0.001,
            0.1,
            0.0001,
            0.0005]

task = SON([
      ('N',1), 
      ('ntrain', 40000),
      ('ntest', 10000),
      ('balance',BALANCE),
      ('universe',SON([('image.ObjectType',SON([('$ne','DCR')]))])),
      ('query',[SON([('image.ObjectType',t)]) for t in TYPES])
      ])

task2 = copy.deepcopy(task)
task2['balance'] = BALANCE2
task3 = copy.deepcopy(task)
task3['balance'] = BALANCE3

config = {

'train_test' : [task,task2,task3]
   
}



