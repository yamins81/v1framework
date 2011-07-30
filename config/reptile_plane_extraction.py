import copy 

from bson import SON

from model_categories import MODEL_CATEGORIES as CAT

            

extraction =  SON([ ('transform_average',SON([('transform_name','translation'),('max',True)])),
                    ('query',SON([('image.model_id',SON([('$in',CAT['reptiles'] + CAT['planes'])])),
                                  ('image.tx',SON([('$exists',False)])),
                                  ('image.s',SON([('$exists',False)])),
                                  ('image.rxy',SON([('$exists',False)])),
                                  ('image.rxz',SON([('$exists',False)])),
                                 ]))
                  ]) 
                
    
config = {
'extractions' : [extraction]
}



