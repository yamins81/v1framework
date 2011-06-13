#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Example Parameters module
"""

#from collections import OrderedDict

from bson import SON

models = [SON([
    ('layers', [SON([('lnorm', SON([
				('inker_shape' , (3,3)),
				('outker_shape', (0,0)),
				('threshold' , 1.0),
				('stretch',1)
				]))]),
				
				SON([('filter',SON([
						('ker_shape',(25,25)),
						('model_name','gridded_gabor'),
						('phases',[0]),
						('norients',16),
						('divfreqs',[2,3,5,8,12,14])
						])),
		
					('activ', SON([
						('min_out' , 0),
						('max_out' , 1),
						])),
					
					('lnorm', SON([
						('inker_shape', (3,3)),
						('outker_shape', (0,0)),
						('threshold', 1.0),
						('stretch',1)
						])),
	
					]),
				 
				 SON([('filter',SON([
				          ('model_name','multiply')
				        ]))
				    ])
				])
				 
	])   
        			    
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
 





