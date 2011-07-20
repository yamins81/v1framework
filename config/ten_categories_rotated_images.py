from math import pi

from starflow.utils import ListUnion

from bson import SON
from model_categories import MODEL_CATEGORIES

MODELS = ListUnion(MODEL_CATEGORIES.values())

NUM_IMAGES = 20
USE_CANONICAL = True

NSEG = 20

base_images = [SON([('model_ids',[m]),
                    ('num_images',NUM_IMAGES),
                    ('use_canonical',USE_CANONICAL),
                    ('generator','renderman'),
                    ('selection','random'),
                    ('ty',SON([('$gt',-.6),('$lt',.6)])),
                    ('tz',SON([('$gt',-.6),('$lt',.6)])),
                    ('ryz',SON([('$gt',2*pi*(ind % NSEG)/NSEG),('$lt',2*pi*((ind % NSEG) + 1)/NSEG)]))
                    ])
               for (ind,m) in enumerate(MODELS)]

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
