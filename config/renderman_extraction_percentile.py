from starflow.utils import ListUnion

import copy 

from bson import SON
            
extraction =  SON([ ('transform_average',SON([('transform_name','translation'),('percentile',[73,80,90,100])])),
                    ('query',SON([('image.bg_id','gray.tdl')
                                 ]))
                  ]) 
                
    
config = {
'extractions' : [extraction]
}



