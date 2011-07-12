#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Example Parameters module
"""

#from collections import OrderedDict

import copy

from bson import SON

import itertools

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
                ('threshold' , 1.0),
                ('stretch',1)
                ]))]),          
            SON([('filter',SON([
                    ('model_name','gridded_gabor'),
                    ('phases',[0]),
                    ('norients',6),
                    ('divfreqs',[2,4,5,7,9,11])
                    ])),
                ('activ', SON([
                    ('min_out' , 0),
                    ('max_out' , 1),
                    ])),
                ('lnorm', SON([
                    ('inker_shape',[3,3]),
                    ('outker_shape',[3,3]),
                    ('threshold', 10.0),
                    ('stretch',.1)
                    ])),
                ('lpool',SON([
                    ('stride',2),
                    ('order',1),
                    ('ker_shape',[9,9])
                    ]))
                ]),
            SON([('filter',SON([
                      ('model_name','really_random'),
                      ('ker_shape',[5,5]),
                      ('num_filters',192)
                    ])),
                ('activ', SON([
                    ('min_out' , 0),
                    ('max_out' , 1),
                    ])),
         
                ])
           ])   
    ])


param_set = [(0,'lnorm',['inker_shape','outker_shape'], [[3,3],[5,5],[7,7],[9,9]]),
             (1,'filter','ker_shape',[[7,7],[9,9],[11,11],[13,13],[17,17],[21,21]]),
            ]

params = list(itertools.product(*[ps[3] for ps in param_set]))

models=[]
for ind1 in range(len(params)):
    for ind2 in range(ind1+1,len(params)):
        p1 = params[ind1]
        p2 = params[ind2]
        model1 =  copy.deepcopy(base_model)
        for (ind,val) in enumerate(p1):
            level,key,atts = param_set[ind][:3]
            if isinstance(atts,str):
                atts = [atts]
            for att in atts:
                model1['layers'][level][key][att] = val
        model2 =  copy.deepcopy(base_model)
        for (ind,val) in enumerate(p2):
            level,key,atts = param_set[ind][:3]
            if isinstance(atts,str):
                atts = [atts]
            for att in atts:
                model2['layers'][level][key][att] = val 
        models.append([model1,model2])


config = {
     'models': models
}
 





