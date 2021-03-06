#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Example Parameters module
"""

#from collections import OrderedDict

import copy
import itertools
from bson import SON

from dbutils import son_escape


base_model = SON([(u'color_space', u'gray'), 
             (u'conv_mode', u'valid'), 
             (u'preproc', SON([(u'max_edge', 150), 
                               (u'lsum_ksize', None), 
                               (u'resize_method', u'bicubic'), 
                               (u'whiten', False)])), 
             (u'layers', [SON([(u'lnorm', SON([(u'inker_shape', [5, 5]), 
                                               (u'outker_shape', [5, 5]), 
                                               (u'threshold', 1.0), 
                                               (u'stretch', 1)]))]), 
                          SON([(u'filter', SON([(u'model_name', u'gridded_gabor'), 
                                                (u'phases', [0]), 
                                                (u'ker_shape', [7, 7]), 
                                                (u'divfreqs', [2, 4, 7, 8, 11]), 
                                                (u'norients', 8)])), 
                               (u'activ', SON([(u'min_out_gen','random'),
                                               (u'max_out', 1)])), 
                               (u'lnorm', SON([(u'inker_shape', [3, 3]), 
                                               (u'outker_shape', [3, 3]), 
                                               (u'threshold', 10.0), 
                                               (u'stretch', 0.1)])), 
                               (u'lpool', SON([(u'order', 2), 
                                               (u'stride', 2), 
                                               (u'ker_shape', [9, 9])]))]),
                          SON([(u'filter',SON([(u'model_name','really_random'),
                                               (u'num_filters',256),
                                               (u'ker_shape',[9,9])])),
                               (u'activ', SON([(u'min_out_gen','random'),
                                               (u'max_out', 1)]))])  
                         ]),
            ])
                        



minmeans1 = [-.3,-.25,-.2,-.15,-.1,-.05]
minmeans2 = [-.3,-.25,-.2,-.15,-.1,-.05]

models=[]
for m1 in minmeans1:
    for m2 in minmeans2:
        model = copy.deepcopy(base_model)
        model['layers'][1]['activ']['min_out_min'] = m1 - .15
        model['layers'][1]['activ']['min_out_max'] = m1 + .15
        model['layers'][2]['activ']['min_out_min'] = m2 - .15
        model['layers'][2]['activ']['min_out_max'] = m2 + .15       
        models.append(model)



config = {
     'models': models
}
 





