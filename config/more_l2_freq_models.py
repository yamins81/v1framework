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
model1 = copy.deepcopy(base_model)
model1['layers'][1]['filter']['ker_shape'] = [9,9]
model1['layers'][1]['filter']['norients'] = 6
model1['layers'][1]['filter']['divfreqs'] = [2,4,6,7,8]

model2 = copy.deepcopy(base_model)
model2['layers'][1]['filter']['ker_shape'] = [7,7]
model2['layers'][1]['filter']['norients'] = 6
model2['layers'][1]['filter']['divfreqs'] = [2,3,4,5,7]

model3 = copy.deepcopy(base_model)
model3['layers'][1]['filter']['ker_shape'] = [13,13]
model3['layers'][1]['filter']['norients'] = 6
model3['layers'][1]['filter']['divfreqs'] = [5,7,9,10,12]


model4 = copy.deepcopy(base_model)
model4['layers'][1]['filter']['ker_shape'] = [21,21]
model4['layers'][1]['filter']['norients'] = 12
model4['layers'][1]['filter']['divfreqs'] = [5,7,9,10,12]


model5 =copy.deepcopy(base_model)
model5['layers'][1]['filter']['ker_shape'] = [21,21]
model5['layers'][1]['filter']['norients'] = 6
model5['layers'][1]['filter']['divfreqs'] = [5,7,9,10,12]

model6 = copy.deepcopy(base_model)
model6['layers'][1]['filter']['ker_shape'] = [21,21]
model6['layers'][1]['filter']['norients'] = 12
model6['layers'][1]['filter']['divfreqs'] = [6,11,13,15,19]

model7 = copy.deepcopy(base_model)
model7['layers'][1]['filter']['ker_shape'] = [21,21]
model7['layers'][1]['filter']['norients'] = 6
model7['layers'][1]['filter']['divfreqs'] = [6,11,13,15,19]


model8 = copy.deepcopy(base_model)
model8['layers'][1]['filter']['ker_shape'] = [21,21]
model8['layers'][1]['filter']['norients'] = 6
model8['layers'][1]['filter']['divfreqs'] = [6,11,13,15,17,19]

models = [base_model,model1,model2,model3,model4,model5,model6,model7,model8]

config = {
     'models': models
}
 





