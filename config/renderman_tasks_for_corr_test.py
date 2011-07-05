import copy 

from bson import SON

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
      

      
tenway_task = SON([
      ('transform_average', SON([('transform_name','translation'),('max',True)])),
      ('N',5), 
      ('ntrain',300),
      ('ntest',150),
      ('universe',SON([('image.bg_id','gray.tdl'),
                       ('$or',combine_ors([[SON([('image.rxy',SON([('$exists',False)]))]),
                                            SON([('image.rxy',SON([('$lt',.3)]))])],
                                           [SON([('image.rxz',SON([('$exists',False)]))]),
                                            SON([('image.rxz',SON([('$lt',.3)]))])]]))
                        ])),
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
        

config = {

'train_test' : [tenway_task
               ]
   
}



