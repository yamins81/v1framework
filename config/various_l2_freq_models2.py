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
model1 = copy.deepcopy(base_model)
model1['layers'][1]['filter']['ker_shape'] = [9,9]
model1['layers'][1]['filter']['norients'] = 4
model1['layers'][1]['filter']['divfreqs'] = [2,4,7,8]

model2 = copy.deepcopy(base_model)
model2['layers'][1]['filter']['ker_shape'] = [9,9]
model2['layers'][1]['filter']['norients'] = 8
model2['layers'][1]['filter']['divfreqs'] = [2,4,7,8]

model3 = copy.deepcopy(base_model)
model3['layers'][1]['filter']['ker_shape'] = [21,21]
model3['layers'][1]['filter']['norients'] = 16
model3['layers'][1]['filter']['divfreqs'] = [2,4,5,8,10,12,15,16]
model3['layers'][2]['filter']['fsample'] = 2
model3['layers'][2]['filter']['osample'] = 2

model4 = copy.deepcopy(base_model)
model4['layers'][1]['filter']['ker_shape'] = [21,21]
model4['layers'][1]['filter']['norients'] = 16
model4['layers'][1]['filter']['divfreqs'] = [2,3,4,5,8,10,12,15,16,18]
model4['layers'][2]['filter']['fsample'] = 2
model4['layers'][2]['filter']['osample'] = 2


model5['layers'][1]['filter']['ker_shape'] = [19,19]
model5['layers'][1]['filter']['norients'] = 16
model5['layers'][1]['filter']['divfreqs'] = [2,4,5,8,10,12,15,16]
model5['layers'][2]['filter']['fsample'] = 2
model5['layers'][2]['filter']['osample'] = 2

model6 = copy.deepcopy(base_model)
model6['layers'][1]['filter']['ker_shape'] = [19,19]
model6['layers'][1]['filter']['norients'] = 16
model6['layers'][1]['filter']['divfreqs'] = [2,3,4,5,8,10,12,15,16,18]
model6['layers'][2]['filter']['fsample'] = 2
model6['layers'][2]['filter']['osample'] = 2

model6a = copy.deepcopy(base_model)
model6a['layers'][1]['filter']['ker_shape'] = [19,19]
model6a['layers'][1]['filter']['norients'] = 16
model6a['layers'][1]['filter']['divfreqs'] = [2,3,4,5,8,10,12,15,16,18]
model6a['layers'][2]['filter']['osample'] = 2


model6b = copy.deepcopy(base_model)
model6b['layers'][1]['filter']['ker_shape'] = [19,19]
model6b['layers'][1]['filter']['norients'] = 16
model6b['layers'][1]['filter']['divfreqs'] = [2,3,4,5,8,10,12,15,16,18]
model6b['layers'][2]['filter']['fsample'] = 2

model7['layers'][1]['filter']['ker_shape'] = [31,31]
model7['layers'][1]['filter']['norients'] = 16
model7['layers'][1]['filter']['divfreqs'] = [2,4,5,8,10,12,15,16]
model7['layers'][2]['filter']['fsample'] = 2
model7['layers'][2]['filter']['osample'] = 2

model8 = copy.deepcopy(base_model)
model8['layers'][1]['filter']['ker_shape'] = [31,31]
model8['layers'][1]['filter']['norients'] = 16
model8['layers'][1]['filter']['divfreqs'] = [2,3,4,5,8,10,12,15,16,18]
model8['layers'][2]['filter']['fsample'] = 2
model8['layers'][2]['filter']['osample'] = 2

model9['layers'][1]['filter']['ker_shape'] = [25,25]
model9['layers'][1]['filter']['norients'] = 16
model9['layers'][1]['filter']['divfreqs'] = [2,4,5,8,10,12,15,16]
model9['layers'][2]['filter']['fsample'] = 2
model9['layers'][2]['filter']['osample'] = 2

model10 = copy.deepcopy(base_model)
model10['layers'][1]['filter']['ker_shape'] = [25,25]
model10['layers'][1]['filter']['norients'] = 16
model10['layers'][1]['filter']['divfreqs'] = [2,3,4,5,8,10,12,15,16,18]
model10['layers'][2]['filter']['fsample'] = 2
model10['layers'][2]['filter']['osample'] = 2

models = [model1,
          model2,
          model3,
          model4,
          model5,
          model6,
          model6a,
          model6b,
          model7,
          model8,
          model9,
          model10]

config = {
     'models': models
}
 





