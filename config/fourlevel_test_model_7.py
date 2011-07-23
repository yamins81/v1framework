#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Example Parameters module
"""

#from collections import OrderedDict

import copy

from bson import SON

model = SON([
    ('color_space','gray'),
    ('conv_mode','same'),
    ('feed_up',True),
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
                    ('mode','same'),
                    ('num_filters',96),
                    ('ker_shape',[5,5])
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
                    ('stride',2),
                    ('order',2),
                    ('ker_shape',[9,9])
                    ]))
                ]),
            SON([('filter',SON([
                    ('model_name','really_random'),
                    ('mode','same'),
                    ('num_filters',96),
                    ('ker_shape',[5,5])
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
                    ('stride',2),
                    ('order',2),
                    ('ker_shape',[9,9])
                    ]))
                ]),
            SON([('filter',SON([
                    ('model_name','really_random'),
                    ('mode','same'),
                    ('num_filters',96),
                    ('ker_shape',[5,5])
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
                    ('stride',1),
                    ('order',2),
                    ('ker_shape',[9,9])
                    ]))
                ]),
            SON([('filter',SON([
                    ('model_name','really_random'),
                    ('mode','same'),
                    ('num_filters',96),
                    ('ker_shape',[5,5])
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
                    ('stride',1),
                    ('order',2),
                    ('ker_shape',[9,9])
                    ]))
                ]),


                

           ])   
    ])


model1 = copy.deepcopy(model)
model1['layers'][2]['lpool']['stride'] = 1

config = {
     'models': [model,model1]
}
 





