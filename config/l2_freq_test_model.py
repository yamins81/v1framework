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
             (u'layers', [SON([(u'lnorm', SON([(u'inker_shape', [9, 9]), 
                                               (u'outker_shape', [9, 9]), 
                                               (u'threshold', 1.0), 
                                               (u'stretch', 1)]))]),
                          SON([(u'filter', SON([(u'model_name', u'gridded_gabor'), 
                                                (u'phases', [0]),
                                                (u'ker_shape', [11, 11]), 
                                                (u'divfreqs', [2, 4, 5, 7, 9]),
                                                (u'norients', 6)])), 
                               (u'activ', SON([(u'min_out', 0), 
                                               (u'max_out', 1)])), 
                               (u'lnorm', SON([(u'inker_shape', [3, 3]), 
                                               (u'outker_shape', [3, 3]), 
                                               (u'threshold', 10.0), 
                                               (u'stretch', 0.10000000000000001)])), 
                               (u'lpool', SON([(u'order', 2),
                                               (u'stride', 2), 
                                               (u'ker_shape', [5, 5])]))]), 
                          SON([(u'filter', SON([(u'model_name', u'freq_uniform'),
                                                (u'ker_shape', [5, 5])])), 
                               (u'activ', SON([(u'min_out', 0),
                                          (u'max_out', 1)]))])
                        ])
        ])

config = {
     'models': [model]
}
 





