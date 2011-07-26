import copy

from math import pi

from starflow.utils import ListUnion

from bson import SON

base_image = SON([('ryz',-pi/2),
               ('tz',1.5),
               ('tx',-3),
               ('bg_id','DH-ITALY10SN.tdl')])
               
    
NSEG = 200

images = []
for ind in range(NSEG):
    im = copy.deepcopy(base_image)
    im['rxy'] = im['bg_phi'] = 2*pi/NSEG * ind
    
    #gorilla
    im['model_id'] = 'MB28626'
    images.append(im)
    
    #deskchair
    im1 = copy.deepcopy(im)
    im1['model_id'] = 'MB29830'
    images.append(im1)
   


config = {'images' : SON([('selection','specific'),
                          ('generator','renderman'),
                          ('specs',images)])}
