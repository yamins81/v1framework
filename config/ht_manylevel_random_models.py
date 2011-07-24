#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Example Parameters module
"""

#from collections import OrderedDict

import copy

from bson import SON

level_0 = SON([('lnorm', SON([
                ('inker_shape' , (9,9)),
                ('outker_shape', (9,9)),
                ('threshold' , 1.0),
                ('stretch',1)
                ]))])
                
level_1 = SON([('filter',SON([
                    ('model_name','really_random'),
                    ('mode','same'),
                    ('num_filters',96),
                    ])),
                ('activ', SON([
                    ('min_out' , 0),
                    ('max_out' , 1),
                    ])),
                ('lnorm', SON([
                    ('threshold', 10.0),
                    ('stretch',.1),
                    ('inker_shape' , (3,3)),
                    ('outker_shape' , (3,3)),
                    ])),
                ('lpool',SON([
                    ('order',2),
                    ('stride',1),
                    ('ker_shape',[5,5])
                    ]))
                ])


base_model = SON([
    ('color_space','gray'),
    ('conv_mode','same'),
    ('feed_up',True),
    ('preproc', SON([
        ('max_edge' , 150),
        ('lsum_ksize' , None),
        ('resize_method','bicubic'),
        ('whiten', False)
    ])),
    ('layers',[level_0])   
    ])

filter_shape = [5,7,9,11]
sizes = [(128,3),(96,4),(77,5),(64,6),
         (86,3),(64,4),(51,5),(43,6)]

models = []
for fs in filter_shape:
        for (n,L) in sizes:
            m = copy.deepcopy(base_model)
            lev = copy.deepcopy(level_1)
            lev['filter']['num_filters'] = n
            lev['filter']['ker_shape'] = [fs,fs]
            m['layers'] += [copy.deepcopy(lev) for ind in range(L)]
            m['layers'][1]['lpool']['ker_shape'] = [9,9]
            m1 = copy.deepcopy(m)
            m['layers'][1]['lpool']['stride'] = 2
            m['layers'][2]['lpool']['stride'] = 2
            m1['layers'][1]['lpool']['stride'] = 2
            models.append(m)
            models.append(m1)

config = {
     'models': models
}
 





