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
                    ('ker_shape',(5,5))
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
model1 = copy.deepcopy(base_model)
model1['layers'][1]['filter']['ker_shape'] = [9,9]
model1['layers'][1]['filter']['norients'] = 4
model1['layers'][1]['filter']['divfreqs'] = [2,4,7,8]

model1a = copy.deepcopy(base_model)
model1a['layers'][2]['filter']['ker_shape'] = [3,3]

model2 = copy.deepcopy(base_model)
model2['layers'][1]['filter']['ker_shape'] = [21,21]
model2['layers'][1]['filter']['norients'] = 12
model2['layers'][1]['filter']['divfreqs'] = [2,4,5,8,10,12,15,16]
model2['layers'][2]['filter']['fsample'] = 2
model2['layers'][2]['filter']['osample'] = 2

model2a = copy.deepcopy(base_model)
model2a['layers'][1]['filter']['ker_shape'] = [21,21]
model2a['layers'][1]['filter']['norients'] = 12
model2a['layers'][1]['filter']['divfreqs'] = [2,5,10,13,16]
model2a['layers'][2]['filter']['osample'] = 2

model3 = copy.deepcopy(base_model)
model3['layers'][1]['filter']['ker_shape'] = [43,43]
model3['layers'][1]['filter']['norients'] = 24
model3['layers'][1]['filter']['divfreqs'] = [4, 6, 8, 9,15,20]
model3['layers'][2]['filter']['osample'] = 2

model3a = copy.deepcopy(base_model)
model3a['layers'][1]['filter']['ker_shape'] = [43,43]
model3a['layers'][1]['filter']['norients'] = 24
model3a['layers'][1]['filter']['divfreqs'] = [5,8,9,15,20,30]
model3a['layers'][2]['filter']['osample'] = 2

#some pooling/norming in l2
model4 = copy.deepcopy(base_model)
model4['layers'][2]['lpool'] = SON([
                    ('order',10),
                    ('stride',2),
                    ('ker_shape',(5,5))
                    ])

model5 = copy.deepcopy(base_model)
model5['layers'][2]['lnorm'] = SON([
                    ('inker_shape', (5,5)),
                    ('outker_shape', (5,5)),
                    ('threshold', 10.0),
                    ('stretch',.1)
                    ])
                    
model6 = copy.deepcopy(base_model)
model6['layers'][2]['lnorm'] = SON([
                    ('inker_shape', (5,5)),
                    ('outker_shape', (5,5)),
                    ('threshold', 10.0),
                    ('stretch',.1)
                    ])      
model6['layers'][2]['lpool'] = SON([
                    ('order',10),
                    ('stride',2),
                    ('ker_shape',(5,5))
                    ])
                    
#different pooling/norming in l1
model7 = copy.deepcopy(base_model)
model7['layers'][1]['lnorm']['inker_shape'] = [3,3]
model7['layers'][1]['lnorm']['outker_shape'] = [3,3]

model8 = copy.deepcopy(base_model)
model8['layers'][1]['lnorm']['stretch'] = 1

model9 = copy.deepcopy(base_model)
model9['layers'][1]['lnorm']['threshold'] = 1

model10 = copy.deepcopy(base_model)
model10['layers'][1]['lpool']['order'] = 10

model11 = copy.deepcopy(base_model)
model11['layers'][1]['lpool']['ker_shape'] = [9,9]

#unif instead of freq unif
model12 = copy.deepcopy(base_model)
model12['layers'][2]['filter']['model_name'] = 'uniform'

models = [model1,
          model1a,
          model2,
          model2a,
          model3,
          model3a,
          model4,
          model5,
          model6,
          model7,
          model8,
          model9,
          model10,
          model11,
          model12]


config = {
     'models': models
}
 





