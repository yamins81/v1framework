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
                    ]))
                ]),
            SON([('filter',SON([
                      ('model_name','freq_uniform'),
                    ])),
                ('activ', SON([
                    ('min_out' , 0),
                    ('max_out' , 1),
                    ])),
         
                ])
           ])   
    ])

l1_norm_shape = [3,5,9]
l1_pool_shape = [3,5,9]
l1_filter_shape = [5,7,9,13,17,21,25,31,37,41]
l2_filter_shape = [3,5,7,9]


l1_filter_params = {5:(4,[2,3,4]),7:(4,[2,3,4,5,6]),9:(6,[2,3,4,5,6,7]),
                    13:(8,[2,3,5,7,9,11]),17:(12,[2,3,5,8,12,14]),21:(16,[2,3,5,9,12,13,15,19]),
                    25:(18,[2,3,5,9,12,13,15,19,23]),31:(18,[2,3,5,9,12,13,15,19,23,25,28]),
                    37:(18,[2,3,5,9,12,13,15,19,23,28]),41:(18,[2,3,5,9,12,13,15,19,23,28])}

def l2_sample(s):
    if s > 8:
        return 2
    else:
        return 1


#different filter shapes
models=[]
for k1 in l1_norm_shape:
    for k2 in l1_pool_shape:
        for k4 in l1_filter_shape:
            for k5 in l2_filter_shape:              
                model =  copy.deepcopy(base_model)
                model['layers'][1]['lnorm']['inker_shape'] = [k1,k1]
                model['layers'][1]['lnorm']['outker_shape'] = [k1,k1]
                model['layers'][1]['lpool']['ker_shape'] = [k2,k2]
                model['layers'][1]['filter']['ker_shape'] = [k4,k4]
                model['layers'][1]['filter']['norients'] = l1_filter_params[k4][0]
                model['layers'][1]['filter']['divfreqs'] = l1_filter_params[k4][1]
                model['layers'][2]['filter']['ker_shape'] = k5
                model['layers'][2]['filter']['osample'] = l2_sample(l1_filter_params[k4][0])
                models.append(model)


config = {
     'models': models
}
 





