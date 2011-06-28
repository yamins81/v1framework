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
                    ('model_name','really_random'),
                    ('num_filters',32)
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
                    ]))
                ]),
           ])   
    ])

l1_norm_shape = [3,5,9]
l1_pool_shape = [3,5,9]
l1_order = [1,2,10]
l1_filter_shape = [5,7,9,11,13,17]


def get_l1_filter_num(p):
    o,df = p
    return o*len(df)

                    
def get_l2_filter_num(p):
    o,df = p
    orn = (o/2 if o > 8 else o)
    return len(df)*(orn**2)


#different filter shapes
models=[]
for k1 in l1_norm_shape:
    for k2 in l1_pool_shape:
        for k3 in l1_order:
            for k4 in l1_filter_shape:
                model =  copy.deepcopy(base_model)
                model['layers'][1]['lnorm']['inker_shape'] = [k1,k1]
                model['layers'][1]['lnorm']['outker_shape'] = [k1,k1]
                model['layers'][1]['lpool']['ker_shape'] = [k2,k2]
                model['layers'][1]['lpool']['order'] =  k3
                model['layers'][1]['filter']['ker_shape'] = [k4,k4]
                models.append(model)


config = {
     'models': models
}
 





