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
                      ('model_name','really_random'),
                    ])),
                ('activ', SON([
                    ('min_out' , 0),
                    ('max_out' , 1),
                    ])),
         
                ])
           ])   
    ])

l1_norm_shape = [3,5]
l1_filter_params = [(4,[2,4,5,7]),(6,[2,4,5,7,9]),(8,[2,4,5,7,9,11])]
l1_filter_shape = [5,7,9,11,13,17]
l2_filter_shape = [3,5]
l2_num_filters = [128,256,384]


#different filter shapes
models=[]
for k1 in l1_norm_shape:
    for k2 in l2_num_filters:
        for k3 in l1_filter_params:
            for k4 in l1_filter_shape:
                for k5 in l2_filter_shape:     
                    model =  copy.deepcopy(base_model)
                    model['layers'][1]['filter']['model_name'] = 'gridded_gabor'
                    model['layers'][1]['filter']['phases'] = [0]
                    model['layers'][1]['lnorm']['inker_shape'] = [k1,k1]
                    model['layers'][1]['lnorm']['outker_shape'] = [k1,k1]
                    model['layers'][1]['filter']['norients'] = k3[0]
                    model['layers'][1]['filter']['divfreqs'] = k3[1]
                    model['layers'][1]['filter']['ker_shape'] = [k4,k4]
                    model['layers'][2]['filter']['ker_shape'] = [k5,k5]
                    model['layers'][2]['filter']['num_filters'] = k2
    
                    models.append(model)
                    
                    model =  copy.deepcopy(base_model)
                    model['layers'][1]['filter']['model_name'] = 'really_random'
                    model['layers'][1]['lnorm']['inker_shape'] = [k1,k1]
                    model['layers'][1]['lnorm']['outker_shape'] = [k1,k1]
                    model['layers'][1]['filter']['num_filters'] = k3[0]*len(k3[1])
                    model['layers'][1]['filter']['ker_shape'] = [k4,k4]
                    model['layers'][2]['filter']['ker_shape'] = [k5,k5]
                    model['layers'][2]['filter']['num_filters'] = k2
    
                    models.append(model)


config = {
     'models': models
}
 





