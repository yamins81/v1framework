#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Example Parameters module
"""

#from collections import OrderedDict

import copy
import itertools
from bson import SON
import config.polygon_task as Images
import config.polygon_correlation_tasks as Tasks
from dbutils import son_escape

model = SON([(u'color_space', u'gray'), 
             (u'conv_mode', u'valid'), 
             (u'preproc', SON([(u'max_edge', 150), 
                               (u'lsum_ksize', None), 
                               (u'resize_method', u'bicubic'), 
                               (u'whiten', False)])), 
             (u'layers', [SON([(u'lnorm', SON([(u'inker_shape', [9, 9]), 
                                               (u'outker_shape', [9, 9]), 
                                               (u'threshold', 1.0), 
                                               (u'stretch', 1)]))]), 
                          SON([(u'filter', SON([(u'model_name', u'gridded_gabor'), 
                                                (u'phases', [0]), 
                                                (u'ker_shape', [13, 13]), 
                                                (u'divfreqs', [2, 4, 7, 8, 11]), 
                                                (u'norients', 7)])), 
                               (u'activ', SON([(u'min_out', 0), 
                                               (u'max_out', 1)])), 
                               (u'lnorm', SON([(u'inker_shape', [9, 9]), 
                                               (u'outker_shape', [9, 9]), 
                                               (u'threshold', 10.0), 
                                               (u'stretch', 0.1)])), 
                               (u'lpool', SON([(u'order', 2), 
                                               (u'stride', 2), 
                                               (u'ker_shape', [5, 5])]))]),
                          SON([(u'filter',SON([(u'model_name','correlation'),
                                               (u'num_filters',64),
                                               (u'random_subset',SON([('const',.5)])),
                                               (u'task',son_escape(Tasks.config['extractions'][0])),
                                               (u'images',son_escape(Images.config['images']))])),
                               (u'activ', SON([(u'min_out', 0), 
                                               (u'max_out', 1)]))])
                         ]),
                        
            ])

config = {
     'models': [model]
}
 





