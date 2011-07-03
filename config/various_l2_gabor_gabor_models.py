#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Example Parameters module
"""

#from collections import OrderedDict

import copy

from bson import SON

base_model = SON([
    ('color_space','gray'),
    ('conv_mode','valid'),
    ('preproc', SON([
        ('max_edge' , 150),
        ('lsum_ksize' , None),
        ('resize_method','bicubic'),
        ('whiten', False)
    ])),
    ('layers', [
            SON([('lnorm', SON([
                ('inker_shape' , (9,9)),
                ('outker_shape', (9,9)),
                ('threshold' , 1.0),
                ('stretch',1)
                ]))]),          
            SON([('filter',SON([
                    ('model_name','gridded_gabor'),
                    ('norients',8),
                    ('divfreqs',[2,4,7,9,11]),
                    ('phases',[0]),
                    ])),
                ('activ', SON([
                    ('min_out' , 0),
                    ('max_out' , 1),
                    ])),
                ('lnorm', SON([
                    ('threshold', 10.0),
                    ('stretch',.1)
                    ])),
                ('lpool',SON([
                    ('stride',2),
                    ('order',2),
                    ('ker_shape',[9,9])
                    ]))
                ]),
            SON([('filter',SON([
                    ('model_name','gridded_gabor'),
                    ('norients',8),
                    ('divfreqs',[2,3,4,7]),
                    ('phases',[0]),
                    ])),
                ('activ', SON([
                    ('min_out' , 0),
                    ('max_out' , 1),
                    ]))
                ])
                
           ])   
    ])

l1_filter_shape = [9,11,13]
l2_filter_shape = [5,9]

#different filter shapes
models=[]
for k1 in l1_filter_shape:
    for k2 in l2_filter_shape:    
        model =  copy.deepcopy(base_model)
        model['layers'][1]['filter']['ker_shape'] = [k1,k1]
        model['layers'][2]['filter']['ker_shape'] = [k2,k2]
        models.append(model)


config = {
     'models': models
}
 





