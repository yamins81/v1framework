#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Example Parameters module
"""

#from collections import OrderedDict

import copy
import itertools
from bson import SON

from dbutils import son_escape


base_model = SON([(u'color_space', u'gray'), 
             (u'conv_mode', u'valid'), 
             (u'preproc', SON([(u'max_edge', 150), 
                               (u'lsum_ksize', None), 
                               (u'resize_method', u'bicubic'), 
                               (u'whiten', False)])), 
             (u'layers', [SON([(u'lnorm', SON([(u'inker_shape', [5, 5]), 
                                               (u'outker_shape', [5, 5]), 
                                               (u'threshold', 1.0), 
                                               (u'stretch', 1)]))]), 
                          SON([(u'filter', SON([(u'model_name', u'gridded_gabor'), 
                                                (u'phases', [0]), 
                                                (u'ker_shape', [13, 13]), 
                                                (u'divfreqs', [2, 4, 7, 8, 11]), 
                                                (u'norients', 8)])), 
                               (u'activ', SON([(u'min_out_gen', 'random'),
                                               (u'max_out', 1)])), 
                               (u'lnorm', SON([(u'inker_shape', [3, 3]), 
                                               (u'outker_shape', [3, 3]), 
                                               (u'threshold', 10.0), 
                                               (u'stretch', 0.1)])), 
                               (u'lpool', SON([(u'order', 2), 
                                               (u'stride', 2), 
                                               (u'ker_shape', [5, 5])]))]),
                          SON([(u'filter',SON([(u'model_name','really_random'),
                                               (u'num_filters',256),
                                               (u'ker_shape',[5,5])])),
                               (u'activ', SON([(u'min_out_gen', 'random'), 
                                               (u'max_out', 1)]))])
                         ]),
            ])
                        



param_set = [(1,'filter','ker_shape',[[5,5],[7,7],[13,13]]),
             (1,'activ','min_out_min',[-.5,-.2]),
             (1,'activ','min_out_max',[.2,.5]),
             (1,'lpool','ker_shape',[[5,5],[7,7],[9,9]]),
             (2,'filter','ker_shape',[[3,3],[5,5],[9,9]]),
             (2,'activ','min_out_min',[-.5,-.2]),
             (2,'activ','min_out_max',[.2,.5])]

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
 





