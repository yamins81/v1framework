#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Example Parameters module
"""

#from collections import OrderedDict

from bson import SON

models = [SON([
    ('layers', [SON([('lnorm', SON([
                ('inker_shape' , (3,3)),
                ('outker_shape', (3,3)),
                ('threshold' , 1.0),
                ('stretch',1),
                ('use_old',True)
                ]))]),
                
            SON([('filter',SON([
                    ('ker_shape',(43,43)),
                    ('model_name','gridded_gabor'),
                    ('phases',[0]),
                    ('norients',24),
                    ('divfreqs',[4, 6, 8, 9,15,20])
                    ])),
    
                ('activ', SON([
                    ('min_out' , 0),
                    ('max_out' , 1),
                    ])),
                
                ('lnorm', SON([
                    ('inker_shape', (3,3)),
                    ('outker_shape', (3,3)),
                    ('threshold', 1.0),
                    ('stretch',1),
                    ('use_old',True)
                    ])),
   
                ('lpool', SON([
                    ('ker_shape', (5,5)),
                    ('stride',2),
                    ('order',2),
                    ])),

                ]),
                         
             SON([('filter',SON([
                      ('model_name','freq_uniform'),
                      ('ker_shape',(5,5)),
                      ('osample',2)
                    ])),
                  ('activ', SON([
                    ('min_out' , 0),
                    ('max_out' , 1),
                    ])),
                ])
           ])   
    ]),
    


                
]

for m in models:
    m['color_space'] ='rgb'
    m['conv_mode'] = 'valid'
    m['preproc'] = SON([
        ('max_edge' , 128),
        ('lsum_ksize' , None),
        ('resize_method','bicubic'),
        ('whiten', False)
    ])
    
config = {
     'models': models
}
 





