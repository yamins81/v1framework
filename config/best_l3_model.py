#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Example Parameters module
"""

#from collections import OrderedDict

from bson import SON

models = [SON([
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
    ]),
]

for m in models:
    m['color_space'] ='rgb'
    m['conv_mode'] = 'valid'
    m['preproc'] = SON([
        ('max_edge' , 150),
        ('lsum_ksize' , None),
        ('resize_method','bicubic'),
        ('whiten', False)
    ])
    
config = {
     'models': models
}
 





