from bson import SON

import model_categories as mc

config = {

'extractions' : [
      SON([('sample_size',2000),
           ('ker_shape',[5,5])])
      SON([('sample_size',2000),
           ('query',SON([('image.model_id',SON([('$in',mc.MODEL_CATEGORIES['faces'])]))])),
           ('ker_shape',[5,5])])
      SON([('sample_size',2000),
           ('query',SON([('image.model_id',SON([('$in',mc.MODEL_CATEGORIES['cars']+mc.MODEL_CATEGORIES['planes'])]))])),
           ('ker_shape',[5,5])])
   ]
   
   
}