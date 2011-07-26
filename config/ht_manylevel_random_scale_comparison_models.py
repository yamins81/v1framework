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
                    ])),
                ('activ', SON([
                    ('min_out' , 0),
                    ('max_out' , 1),
                    ])),
                ('lpool',SON([
                    ('order',2),
                    ('stride',1),
                    ]))
                ])

base_model = SON([
    ('color_space','gray'),
    ('conv_mode','same'),
    ('feed_up',True),
    ('preproc', SON([
        ('lsum_ksize' , None),
        ('resize_method','bicubic'),
        ('whiten', False)
    ])),
    ('layers',[level_0])   
    ])

filter_shape = [3,5,7,9]
sizes = [(256,1),(128,2),(85,3),(64,4),(51,5)]
me = [64,256]

models = []
for fs in filter_shape:
    for (n,L) in sizes:
        for sz in me:
            m = copy.deepcopy(base_model)
            m['preproc']['max_edge'] = sz
            lev = copy.deepcopy(level_1)
            lev['filter']['num_filters'] = n
            lev['filter']['ker_shape'] = [fs,fs]
            lev['lpool']['ker_shape'] = [fs,fs]
            m['layers'] += [copy.deepcopy(lev) for ind in range(L)]
            models.append(m)

    

config = {
     'models': models
}
 





