from bson import SON

import model_categories as mc

import itertools

import copy

def combine_dicts(d):
    y = copy.deepcopy(d[0])
    for dd in d[1:]:
        y.update(dd)
    return y
    
def combine_ors(ors):
    return map(combine_dicts,itertools.product(ors))
    
    
config = {


'train_test' : [

   SON([
      ('transform_average', SON([('transform_name','translation'),('max',True)])),
      ('N',1), 
      ('ntrain',80),
      ('ntest',40),
      ('universe',SON([('image.bg_id','gray.tdl'),
                       ('$or',combine_ors([[SON([('image.rxy',SON([('$exists',False)]))]),
                                            SON([('image.rxy',SON([('$lt',.1)]))])],
                                           [SON([('image.rxz',SON([('$exists',False)]))]),
                                            SON([('image.rxz',SON([('$lt',.1)]))])]]))
                      ])),
      ('query',[SON([('image.model_id',SON([('$in',mc.MODEL_CATEGORIES['cars'])]))]),
                SON([('image.model_id',SON([('$in',mc.MODEL_CATEGORIES['planes'])]))]),
               ])
      ]), 
   SON([
      ('transform_average', SON([('transform_name','translation'),('max',True)])),
      ('N',1), 
      ('ntrain',80),
      ('ntest',40),
      ('universe',SON([('image.bg_id','gray.tdl'),
                       ('$or',combine_ors([[SON([('image.rxy',SON([('$exists',False)]))]),
                                            SON([('image.rxy',SON([('$lt',.1)]))])],
                                           [SON([('image.rxz',SON([('$exists',False)]))]),
                                            SON([('image.rxz',SON([('$lt',.1)]))])],
                                           [SON([('image.ryz',SON([('$exists',False)]))]),
                                            SON([('image.ryz',SON([('$lt',.1)]))])]]))
                      ])),
      ('query',[SON([('image.model_id',SON([('$in',mc.MODEL_CATEGORIES['cars'])]))]),
                SON([('image.model_id',SON([('$in',mc.MODEL_CATEGORIES['planes'])]))]),
               ])
      ]),

   SON([
      ('transform_average', SON([('transform_name','translation'),('max',True)])),
      ('N',1), 
      ('ntrain',80),
      ('ntest',40),
      ('universe',SON([('image.bg_id','gray.tdl'),
                       ('$or',combine_ors([[SON([('image.tx',SON([('$exists',False)]))]),
                                            SON([('image.tx',SON([('$lt',.2),('$gt',-.2)]))])],
                                           [SON([('image.ty',SON([('$exists',False)]))]),
                                            SON([('image.ty',SON([('$lt',.2),('$gt',-.2)]))])]]))
                      ])),
      ('query',[SON([('image.model_id',SON([('$in',mc.MODEL_CATEGORIES['cars'])]))]),
                SON([('image.model_id',SON([('$in',mc.MODEL_CATEGORIES['planes'])]))]),
               ])
      ]),
   SON([
      ('transform_average', SON([('transform_name','translation'),('max',True)])),
      ('N',1), 
      ('ntrain',80),
      ('ntest',40),
      ('universe',SON([('image.bg_id','gray.tdl'),
                       ('$or',[SON([('config.image.sx',SON([('$exists',False)]))]),
                               SON([('$where',"(this.config.image.sx/this.config.image.sy < 1.1) && (this.config.image.sx/this.config.image.sy > 1/1.1)")])
                              ])
                      ])),
      ('query',[SON([('image.model_id',SON([('$in',mc.MODEL_CATEGORIES['cars'])]))]),
                SON([('image.model_id',SON([('$in',mc.MODEL_CATEGORIES['planes'])]))]),
               ])
      ]),


   ]
   
   
}



