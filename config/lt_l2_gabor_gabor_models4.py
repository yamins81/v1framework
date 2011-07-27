#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Example Parameters module
"""

#from collections import OrderedDict

from math import pi

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
                    ('mode','same'),
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
                      ('model_name','random_gabor'),
                      ('mode','same'),
                      ('z_envelope_center',SON([('min',.25),('max',.75)])),
                      ('frequency_range',[0,.5]),
                      ('orient_ranges',[[pi/2,pi/2],[0,0]])
                    ])),
                ('activ', SON([
                    ('min_out' , 0),
                    ('max_out' , 1),
                    ])),
         
                ])
           ])   
    ])

l1_norm_shape = [3]
l1_filter_params = [(6,[2,4,5,7,9])]
l1_filter_shape = [9]
l2_filter_shape = [9]
l2_num_filters = [256]
orient_ranges = []
frequency_ranges = [[.2,.5],[.3,.2],[.1,.5],[[.2,.5],[.2,.5],[.2,.5]]]


#different filter shapes
models=[]
for k1 in l1_norm_shape:
    for k2 in l2_num_filters:
        for k3 in l1_filter_params:
            for k4 in l1_filter_shape:
                for k5 in l2_filter_shape:  
                    for k6 in frequency_ranges:
                        model =  copy.deepcopy(base_model)
                        model['layers'][1]['lnorm']['inker_shape'] = [k1,k1]
                        model['layers'][1]['lnorm']['outker_shape'] = [k1,k1]
                        model['layers'][1]['filter']['norients'] = k3[0]
                        model['layers'][1]['filter']['divfreqs'] = k3[1]
                        model['layers'][1]['filter']['ker_shape'] = [k4,k4]
                        model['layers'][2]['filter']['ker_shape'] = [k5,k5]
                        model['layers'][2]['filter']['num_filters'] = k2
                        model['layers'][2]['filter']['frequency_range'] = k6
                        models.append(model)


config = {
     'models': models
}
 





