from bson import SON

import model_categories as mc

config = {

'extractions' : [
      SON([('sample_size',2500),
           ('ker_shape',[3,3])]),
      SON([('sample_size',2500),
           ('ker_shape',[5,5])]),
   ]
   
   
}