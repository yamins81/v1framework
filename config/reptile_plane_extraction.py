import copy 

from bson import SON

from model_categories import MODEL_CATEGORIES as CAT


extraction =  SON([ ('transform_average',SON([('transform_name','translation'),('max',True)])),
                    ('query',SON([('image.model_id',SON([('$in',CAT['reptiles'] + CAT['planes'])]))]))
                  ]) 
                
    
config = {
'extractions' : [extraction]
}



