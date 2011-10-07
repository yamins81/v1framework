import copy 

from bson import SON

import model_categories as mc
      
TYPES =  [u'Car',u'Container', u'Plane', []]

task = SON([
      ('N',10), 
      ('ntrain', 500),
      ('balance',[.3,.25]),
      ('ntest', 150),
      ('universe',SON([('image.ObjectType',SON([('$ne','DCR')]))])),
      ('query',[SON([('image.ObjectType',t)]) for t in [u'Car',u'Plane', []]])
      ])

task_b = SON([
          ('N',10),
          ('ntrain', 1000),
          ('ntest', 150),
          ('balance',[.02,.005,.01]),
          ('universe',SON([('image.ObjectType',SON([('$ne','DCR')]))])),
          ('query',[SON([('image.ObjectType',t)]) for t in [u'Car',u'Container',u'Plane', []]])
          ])
        
config = {

'train_test' : [task,task_b]
   
}



