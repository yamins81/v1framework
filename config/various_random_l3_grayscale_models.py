#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Example Parameters module
"""

#from collections import OrderedDict

import copy

from bson import SON

base_model = SON([
    ('layers', [SON([('lnorm', SON([
                ('inker_shape' , (9,9)),
                ('outker_shape', (9,9)),
                ('threshold' , 1.0),
                ('stretch',1)
                ]))]),
            SON([('filter',SON([
                    ('ker_shape',(3,3)),
                    ('model_name','really_random'),
                    ('num_filters',16)
                    ])),
                ('activ', SON([
                    ('min_out' , 0),
                    ('max_out' , None),
                    ])),
                ('lnorm', SON([
                    ('inker_shape', (9,9)),
                    ('outker_shape', (9,9)),
                    ('threshold', 10.0),
                    ('stretch',.1)
                    ])),
                ('lpool',SON([
                    ('order',2),
                    ('stride',2),
                    ('ker_shape',(5,5))
                    ]))
                ]),
            SON([('filter',SON([
                    ('model_name','really_random'),
                    ('num_filters',32),
                    ('ker_shape',(3,3)),
                    ])),
                ('activ', SON([
                    ('min_out' , 0),
                    ('max_out' , None),
                    ])),
                ('lnorm', SON([
                    ('inker_shape', (5,5)),
                    ('outker_shape', (5,5)),
                    ('threshold', 10.0),
                    ('stretch',.1),
                    ('remove_mean',True)
                    ])),           
                ('lpool',SON([
                    ('order',10),
                    ('stride',2),
                    ('ker_shape',(5,5))
                    ]))         
                ]),
            SON([('filter',SON([
                    ('model_name','really_random'),
                    ('num_filters',128),
                    ('ker_shape',(3,3)),
                    ])),
                ('activ', SON([
                    ('min_out' , None),
                    ('max_out' , 1),
                    ])),
                ('lnorm', SON([
                    ('inker_shape', (3,3)),
                    ('outker_shape', (3,3)),
                    ('threshold', 10.0),
                    ('stretch',.1),
                    ])),
                ('lpool',SON([
                    ('order',2),
                    ('stride',2),
                    ('ker_shape',(9,9))
                    ]))
                ])
           ])    
    ])


for m in [base_model]:
    m['color_space'] ='gray'
    m['conv_mode'] = 'valid'
    m['preproc'] = SON([
        ('max_edge' , 150),
        ('lsum_ksize' , None),
        ('resize_method','bicubic'),
        ('whiten', False)
    ])
    
l0_norm_kers = [3,9]
l1_filter_kers = [3,9]
l1_norm_kers = [3,9]
l1_pool_kers = [3,9]

models = []
for v1 in l0_norm_kers:
    for v2 in l1_filter_kers:
        for v3 in l1_norm_kers:
            for v4 in l1_pool_kers:
                m = copy.deepcopy(base_model)
                m['layers'][0]['lnorm']['inker_shape'] = [v1,v1]
                m['layers'][0]['lnorm']['outker_shape'] = [v1,v1]
                m['layers'][1]['filter']['ker_shape'] = [v2,v2]
                m['layers'][1]['lnorm']['inker_shape'] = [v3,v3]
                m['layers'][1]['lnorm']['outker_shape'] = [v3,v3]
                m['layers'][1]['lpool']['ker_shape'] = [v4,v4]
                models.append(m)

config = {
     'models': models
}
 





