from math import pi

from starflow.utils import ListUnion

from bson import SON
from model_categories import MODEL_CATEGORIES

MODELS = ListUnion(MODEL_CATEGORIES.values())

config = {'images' : 
          SON([('model_ids',MODELS),
               ('bg_query',SON([("type","3d hdr")])),
               ('num_images',10000),
               ('use_canonical',True),
               ('ty',SON([('$gt',1),('$lt',1)])),
               ('s',SON([('$gt',.5),('$lt',2)])),
               ('tz',SON([('$gt',-1),('$lt',1)])),
               ('rxy',SON([('$gt',-pi/2),('$lt',pi/2)])),
               ('rxz',SON([('$gt',-pi/2),('$lt',pi/2)])),
               ('ryz',SON([('$gt',-pi),('$lt',pi)])),
               ('generator','renderman'),
               ('selection','random')
          ])
}
