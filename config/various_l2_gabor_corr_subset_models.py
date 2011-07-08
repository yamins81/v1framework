#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Example Parameters module
"""

#from collections import OrderedDict
import copy
import itertools
from bson import SON
import config.ten_categories_images as Images
import config.renderman_correlation_tasks as Tasks
from dbutils import son_escape

import config.various_l1_gabors as l1_models

corr_layer =  SON([(u'filter',SON([(u'model_name','correlation'),
                                               (u'random_subset',SON([('const',.5)])),
                                               (u'num_filters',256),
                                               (u'task',son_escape(Tasks.config['extractions'][0])),
                                               (u'images',son_escape(Images.config['images']))])),
                                   (u'activ', SON([(u'min_out', 0), 
                                                   (u'max_out', 1)]))])


models = []
for M in l1_models.config['models']:
    m = copy.deepcopy(M)
    m['layers'].append(corr_layer)
    models.append(m)


config = {
     'models': models
}
 





