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

filter_shape = [7,9,11,13]
sizes = [(192,2),(128,3),(96,4),(77,5),(64,6)]
gabors = {192:(16,[2,3,4,5,6,7,8,9,10,11,12,13]),
          128:(16,[2,4,5,7,9,11,13,15]),
          96:(16,[2,4,7,9,11,13]),
          77:(11,[2,4,7,9,11,13,15]),
          64:(8,[2,4,5,7,9,11,13,15])}

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
        
        m2 = copy.deepcopy(m)
        m3 = copy.deepcopy(m1)
        for M in [m2,m3]:
            M['layers'][1]['filter'].pop('num_filters')
            M['layers'][1]['filter']['model_name'] = 'gridded_gabor'
            M['layers'][1]['filter']['norients'] = gabors[n][0]
            M['layers'][1]['filter']['divfreqs'] = gabors[n][1]
            M['layers'][1]['filter']['phases'] = [0]
        models.append(m2)
        models.append(m3)        


config = {
     'models': models
}
 





