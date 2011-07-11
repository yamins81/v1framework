#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Example Parameters module
"""

#from collections import OrderedDict
import copy
import itertools
from bson import SON
import config.ten_categories_images as Images
import config.renderman_correlation_tasks2 as Tasks
from dbutils import son_escape

import config.ht_l1_gabor_models_for_corr as l1_models

corr_layer =  SON([(u'filter',SON([(u'model_name','correlation'),
                                   (u'random_subset',SON([('const',.5)])),
                                   (u'images',son_escape(Images.config['images']))])),
                   (u'activ', SON([(u'min_out', 0), 
                                   (u'max_out', 1)]))])


models = []
for M in l1_models.config['models']:
    for num_filters in [256,384]:
        for task in Tasks.config['extractions']:
            m = copy.deepcopy(M)
            m['layers'].append(copy.deep_copy(corr_layer))
            m['layers'][2]['filter']['task'] = son_escape(task)
            m['layers'][2]['filter']['num_filters'] = num_filters
            models.append(m)
    


config = {
     'models': models
}
 





