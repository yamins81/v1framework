from math import pi

from starflow.utils import ListUnion

from bson import SON
from model_categories import MODEL_CATEGORIES

MODELS = MODEL_CATEGORIES['faces']

NUM_IMAGES = 500
USE_CANONICAL = True

base_images = [
          #just the images
          SON([('model_ids',MODELS),
               ('use_canonical',USE_CANONICAL),
               ('generator','renderman'),
               ('selection','gridded')
          ]),
          #translation alone
          SON([('model_ids',MODELS),
               ('num_images',NUM_IMAGES),
               ('use_canonical',USE_CANONICAL),
               ('generator','renderman'),
               ('selection','random'),
               ('ty',SON([('$gt',-.6),('$lt',.6)])),
               ('tz',SON([('$gt',-.6),('$lt',.6)])),
          ]), 
          #in-plane rotation alone
          SON([('model_ids',MODELS),
               ('num_images',NUM_IMAGES),
               ('use_canonical',USE_CANONICAL),
               ('generator','renderman'),
               ('selection','random'),
               ('ryz',SON([('$gt',-pi),('$lt',pi)])),
          ]),
          #out-of-plane rotation alone
          SON([('model_ids',MODELS),
               ('num_images',NUM_IMAGES),
               ('use_canonical',USE_CANONICAL),
               ('generator','renderman'),
               ('selection','random'),
               ('rxy',SON([('$gt',-pi/2),('$lt',pi/2)])),
               ('rxz',SON([('$gt',-pi/2),('$lt',pi/2)])),
          ]),
          #all rotation
          SON([('model_ids',MODELS),
               ('num_images',NUM_IMAGES),
               ('use_canonical',USE_CANONICAL),
               ('generator','renderman'),
               ('selection','random'),
               ('rxy',SON([('$gt',-pi/2),('$lt',pi/2)])),
               ('rxz',SON([('$gt',-pi/2),('$lt',pi/2)])),
               ('ryz',SON([('$gt',-pi),('$lt',pi)])),
          ]),
          #scale alone
          SON([('model_ids',MODELS),
               ('num_images',NUM_IMAGES),
               ('use_canonical',USE_CANONICAL),
               ('generator','renderman'),
               ('selection','random'),
               ('s',SON([('$gt',.5),('$lt',2)])),
          ]),
          #translation + inplane rotation
          SON([('model_ids',MODELS),
               ('num_images',NUM_IMAGES),
               ('use_canonical',USE_CANONICAL),
               ('generator','renderman'),
               ('selection','random'),
               ('ty',SON([('$gt',-.6),('$lt',.6)])),
               ('tz',SON([('$gt',-.6),('$lt',.6)])),
               ('ryz',SON([('$gt',-pi),('$lt',pi)])),
          ]),
          #translation + all rotation
          SON([('model_ids',MODELS),
               ('num_images',NUM_IMAGES),
               ('use_canonical',USE_CANONICAL),
               ('generator','renderman'),
               ('selection','random'),
               ('ty',SON([('$gt',-.6),('$lt',.6)])),
               ('tz',SON([('$gt',-.6),('$lt',.6)])),
               ('rxy',SON([('$gt',-pi/2),('$lt',pi/2)])),
               ('rxz',SON([('$gt',-pi/2),('$lt',pi/2)])),
               ('ryz',SON([('$gt',-pi),('$lt',pi)])),
          ]),
          #inplane rotation + scale
          SON([('model_ids',MODELS),
               ('num_images',NUM_IMAGES),
               ('use_canonical',USE_CANONICAL),
               ('generator','renderman'),
               ('selection','random'),
               ('s',SON([('$gt',.5),('$lt',2)])),
               ('ryz',SON([('$gt',-pi),('$lt',pi)])),
          ]),
          #all rotation + scale
          SON([('model_ids',MODELS),
               ('num_images',NUM_IMAGES),
               ('use_canonical',USE_CANONICAL),
               ('generator','renderman'),
               ('selection','random'),
               ('s',SON([('$gt',.5),('$lt',2)])),
               ('rxy',SON([('$gt',-pi/2),('$lt',pi/2)])),
               ('rxz',SON([('$gt',-pi/2),('$lt',pi/2)])),
               ('ryz',SON([('$gt',-pi),('$lt',pi)])),
          ]),
         #everything 
          SON([('model_ids',MODELS),
               ('num_images',10*NUM_IMAGES),
               ('use_canonical',USE_CANONICAL),
               ('generator','renderman'),
               ('selection','random'),
               ('ty',SON([('$gt',-.6),('$lt',.6)])),
               ('tz',SON([('$gt',-.6),('$lt',.6)])),
               ('s',SON([('$gt',.5),('$lt',2)])),
               ('rxy',SON([('$gt',-pi/2),('$lt',pi/2)])),
               ('rxz',SON([('$gt',-pi/2),('$lt',pi/2)])),
               ('ryz',SON([('$gt',-pi),('$lt',pi)])),
          ]),

]
import copy

imagesets = []

#gray background
for m in base_images:
    mc = copy.deepcopy(m)
    mc['bg_ids'] = ['gray.tdl']
    imagesets.append(mc)
    
#3d hdr backgrounds
for m in base_images:
    mc = copy.deepcopy(m)
    mc['bg_query'] = SON([('type','3d hdr')])
    mc['bg_phi'] = SON([('$gt',-pi),('$lt',pi)])
    mc['bg_psi'] = SON([('$gt',-pi),('$lt',pi)])
    imagesets.append(mc)

config = {'images' : imagesets
}
