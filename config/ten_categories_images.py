from starflow.utils import ListUnion

from bson import SON
from model_categories import MODEL_CATEGORIES

MODELS = ListUnion(MODEL_CATEGORIES.values())

config = {'images' : 
          SON([('model_ids',MODELS),
               ('bg_query',SON([("type","3d hdr")])),
               ('num_images',100),
               ('use_canonical',True),
               ('ty',SON([('$gt',-.1),('$lt',.1)])),
               ('generator','renderman'),
               ('selection','random')
          ])
}