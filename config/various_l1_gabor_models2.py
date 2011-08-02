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
                    ('order',2),
                    ('ker_shape',[9,9])
                    ]))
                ])
           ])   
    ])

param_set = [(0,'lnorm',['inker_shape','outker_shape'], [[5,5]]),
             (1,'lnorm',['inker_shape','outker_shape'], [[3,3]]),
             (1,'filter','ker_shape',[[9,9],[11,11],[13,13],[17,17],[21,21],[25,25],[31,31],[41,41]]),
             (1,'filter','norients',[16]),
             (1,'filter','divfreqs',[[2,4,7,9,11,15]]),
             (1,'activ','min_out',[-.5,0,.5]),
             (1,'lpool','ker_shape',[[9,9]])]

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
 





