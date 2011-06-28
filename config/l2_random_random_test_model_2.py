#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Example Parameters module
"""

#from collections import OrderedDict

import copy

from bson import SON

model = SON([(u'color_space', u'gray'), 
             (u'conv_mode', u'valid'), 
             (u'preproc', SON([(u'max_edge', 150), 
             (u'lsum_ksize', None), 
             (u'resize_method', u'bicubic'), 
             (u'whiten', False)])), 
             (u'layers', [SON([(u'lnorm', SON([(u'inker_shape', [9, 9]), (u'outker_shape', [9, 9]), (u'threshold', 1.0), (u'stretch', 1)]))]),
                          SON([(u'filter', SON([(u'model_name', u'really_random'), (u'num_filters', 64), (u'ker_shape', [7, 7])])), 
                               (u'activ', SON([(u'min_out', 0), (u'max_out', 1)])), 
                               (u'lnorm', SON([(u'threshold', 10.0), (u'stretch', 0.10000000000000001), (u'inker_shape', [3, 3]), (u'outker_shape', [3, 3])])), 
                               (u'lpool', SON([(u'stride', 2), (u'ker_shape', [9, 9])]))]),
                          SON([(u'filter', SON([(u'model_name', u'really_random'), (u'num_filters', 512), (u'ker_shape', [5, 5])])), 
                               (u'activ', SON([(u'min_out', 0), (u'max_out', 1)]))])])
        ])

config = {
     'models': [model]
}
 





