#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Example Parameters module
"""

#from collections import OrderedDict

from bson import SON

models = [SON([
    ('layers', [SON([('lnorm', SON([
				('inker_shape' , (3,3)),
				('threshold' , 1.0),
				]))]),
				
            SON([('filter',SON([
					('ker_shape',(13,13)),
					('model_name','gridded_gabor'),
					('phases',[0]),
					('norients',8),
					('divfreqs',[2,5,8,12])
					])),
	
				('activ', SON([
					('min_out' , 0),
					('max_out' , 1),
					])),
				
				('lnorm', SON([
					('inker_shape', (3,3)),
					('threshold', 1.0),
					])),
				
				('lpool',SON([
					('order',2),
					('stride',2),
					('ker_shape',(3,3))
					]))
			    
			    ]),
            SON([('filter',SON([
                    ('model_name','really_random'), 
					('num_filters',32),
					('ker_shape',(5,5)),
					])),
	
				('activ', SON([
					('min_out' , 0),
					('max_out' , 1),
					])),
				
				('lnorm', SON([
					('inker_shape', (3,3)),
					('threshold', 1.0),
					])),
				
				('lpool',SON([
					('order',2),
					('stride',2),
					('ker_shape',(3,3))
					]))
			    
			    ])
			
		   ])    
    ]),
    
SON([
    ('layers', [SON([('lnorm', SON([
				('inker_shape' , (3,3)),
				('threshold' , 1.0),
				]))]),
				
            SON([('filter',SON([
					('ker_shape',(9,9)),
					('model_name','gridded_gabor'),
					('phases',[0]),
					('norients',8),
					('divfreqs',[2,5,8,12])
					])),
	
				('activ', SON([
					('min_out' , 0),
					('max_out' , 1),
					])),
				
				('lnorm', SON([
					('inker_shape', (3,3)),
					('threshold', 1.0),
					])),
				
				('lpool',SON([
					('order',2),
					('stride',2),
					('ker_shape',(3,3))
					]))
			    
			    ]),
            SON([('filter',SON([
                    ('model_name','really_random'), 
					('num_filters',16),
					('ker_shape',(5,5)),
					])),
	
				('activ', SON([
					('min_out' , 0),
					('max_out' , 1),
					])),
				
				('lnorm', SON([
					('inker_shape', (3,3)),
					('threshold', 1.0),
					])),
				
				('lpool',SON([
					('order',2),
					('stride',2),
					('ker_shape',(3,3))
					]))
			    
			    ])
			
		   ])    
    ])
]

for m in models:
	m['color_space'] ='rgb'
	m['conv_mode'] = 'valid'
	m['preproc'] = SON([
		('max_edge' , 64),
		('lsum_ksize' , None),
		('resize_method','bicubic'),
		('whiten', False)
	])
    
config = {
     'models': models
}
 





