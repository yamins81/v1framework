#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Example Parameters module
"""

#from collections import OrderedDict

import copy
import itertools
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

param_set = [(0,'lnorm',['inker_shape','outker_shape'], [[3,3],[9,9]]),
             (1,'lnorm',['inker_shape','outker_shape'], [[3,3],[5,5],[7,7]]),
             (1,'filter','ker_shape',[[7,7],[9,9],[11,11],[13,13]]),
             (1,'lpool','ker_shape',[[5,5],[9,9]]),
             (2,'filter','ker_shape',[[3,3],[5,5],[7,7],[9,9]])]

params = itertools.product(*[ps[3] for ps in param_set])

models=[]
for p in params:
    model =  copy.deepcopy(base_model)
    for (ind,val) in enumerate(p):
        level,key,atts = param_set[ind][:3]
        if isinstance(atts,str):
            atts = [atts]
        for att in atts:
            model['layers'][level][key][att] = val        
    models.append(model)


config = {
     'models': models
}
 





