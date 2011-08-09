from math import pi

from starflow.utils import ListUnion

from bson import SON
from model_categories import MODEL_CATEGORIES

MODELS = MODEL_CATEGORIES['reptiles'] + MODEL_CATEGORIES['planes']

NUM_IMAGES = 2000
USE_CANONICAL = True

base_images = [
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
]
import copy

imagesets = []

#gray background
for m in base_images:
    mc = copy.deepcopy(m)
    mc['bg_ids'] = ['gray.tdl']
    mc['res'] = 150
    imagesets.append(mc)
    
config = {'images' : imagesets
}
