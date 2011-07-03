import copy 

from bson import SON
import math
import model_categories as mc

def combine_ors(ors):
    import copy
    import itertools
    def combine_dicts(d):
        y = copy.deepcopy(d[0])
        for dd in d[1:]:
            y.update(dd)
        return y
    return map(combine_dicts,itertools.product(*ors))
      

NSEG = 4
      
queries = [SON([('$or',combine_ors([[SON([('image.model_id',o),
                                          ('image.ryz',SON([('$gt',2*math.pi*(ind % NSEG)/NSEG),
                                                            ('$lt',2*math.pi*(ind % NSEG + 1)/NSEG)]))]) 
                       for (ind,o) in enumerate(mc.MODEL_CATEGORIES[cat])],
                      [SON([('image.s',SON([('$exists',False)]))]),
                       SON([('image.s',SON([('$lt',1.25),('$gt',1/1.25)]))])]])),
          ('image.rxy',SON([('$exists',False)])),
          ('image.rxz',SON([('$exists',False)]))]) for cat in ['cars',
                                                      'planes',
                                                      'boats',
                                                      'faces',
                                                      'chair',
                                                      'table',
                                                      'plants',
                                                      'reptiles',
                                                      'cats_and_dogs']
]                                
 


task = SON([
      ('transform_average', SON([('transform_name','translation'),('max',True)])),
      ('N',3), 
      ('ntrain',270),
      ('ntest',180),
      ('universe',SON([('image.bg_id','gray.tdl')])),
      ('query',queries)
      ])
    


config = {
'train_test' : [task]
}



