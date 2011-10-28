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

BALANCE = [0.0004,
            0.03,
            0.002,
            0.001,
            0.0001,
            0.001,
            0.006,
            0.0001,
            0.0004]

task = SON([
      ('N',1), 
      ('ntrain', 40000),
      ('ntest', 10000),
      ('balance',BALANCE),
      ('universe',SON([('image.ObjectType',SON([('$ne','DCR')]))])),
      ('query',[SON([('image.ObjectType',t)]) for t in TYPES])
      ])
        
config = {

'train_test' : [task]
   
}



