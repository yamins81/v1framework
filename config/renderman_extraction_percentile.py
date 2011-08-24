from starflow.utils import ListUnion

import copy 

from bson import SON

from model_categories import MODEL_CATEGORIES as CAT

all_models = ListUnion(CAT.values())
            

extraction =  SON([ ('transform_average',SON([('transform_name','translation'),('percentile',[73,80,90,100])])),
                    ('query',SON([('image.model_id',SON([('$in',all_models)])),
                                  ('image.bg_id','gray.tdl')
                                 ]))
                  ]) 
                
    
config = {
'extractions' : [extraction]
}



