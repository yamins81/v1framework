import copy 

from bson import SON

import model_categories as mc
      
NTRAIN = 
NTEST = 

TYPES =  [u'Car',u'Container', u'Plane', u'']

task = SON([
      ('N',10), 
      ('ntrain', NTRAIN),
      ('ntest', NTEST),
      ('universe',SON([('image.ObjectType',SON([('$ne','DCR')]))])),
      ('query',[SON([('image.ObjectType',t)]) for t in TYPES])
      ])
        
task_b = copy.deepcopy(task)
task_b['balance'] = [.02,.005,.01]
 
config = {

'train_test' : [task,task_b]
   
}



