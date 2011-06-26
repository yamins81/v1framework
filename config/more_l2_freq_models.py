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
                    ('ker_shape',[11,11]),
                    ('divfreqs',[2,4,5,7,9]),
                    ('norients',6)
                    ])),
                ('activ', SON([
                    ('min_out' , 0),
                    ('max_out' , 1),
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
                    ('ker_shape',(9,9))
                    ]))
                ]),
            SON([('filter',SON([
                      ('model_name','freq_uniform'),
                      ('ker_shape',(5,5)),
                    ])),
                ('activ', SON([
                    ('min_out' , 0),
                    ('max_out' , 1),
                    ])),
         
                ])
           ])   
    ])

#different filter shapes
#different filter shapes
models=[]
for k1 in [7,9,11,13,15]:
    for k2 in [3,5,7,9]:
        model =  copy.deepcopy(base_model)
        model['layers'][1]['filter']['ker_shape'] = [k1,k1]
        model['layers'][2]['filter']['ker_shape'] = [k2,k2]
        models.append(model)


config = {
     'models': models
}
 





