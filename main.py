#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import cPickle

from bson import SON

from starflow.protocols import protocolize, actualize
from starflow.utils import activate

from dbutils import get_config_string,  reach_in, get_filename, DBAdd, createCertificateDict, son_escape, do_initialization

from v1like_extract import v1_feature_extraction_protocol, v1_initialize, get_config

from traintest import train_test

@protocolize()
def test_extract_cairo(depends_on = '../config/config_cairo_test_extract.py'):
    v1_feature_extraction_protocol(depends_on,write=True)

@protocolize()
def test_extract_really_random(depends_on = '../config/config_testreallyrandom.py'):
    v1_feature_extraction_protocol(depends_on,write=True)
    
@protocolize()
def test_extract_random_gabor(depends_on = '../config/config_testrandomgabor.py'):
    v1_feature_extraction_protocol(depends_on,write=True)

@protocolize()
def pixel_trans_extraction(depends_on = '../config/config_pixel_trans.py'):
    v1_feature_extraction_protocol(depends_on,write=True)

@protocolize()
def pixel_trans_evaluation(depends_on = ('../config/config_pixel_trans.py','../config/config_pixel_trans_evaluation.py')):
    D = v1_evaluation_protocol(depends_on[1],depends_on[0])    
    actualize(D)
  
@protocolize()
def pixel_trans_large_extraction(depends_on = '../config/config_pixel_trans_large.py'):
    v1_feature_extraction_protocol(depends_on,write=True)
  
  
@protocolize()
def pixel_trans_large_evaluation(depends_on = ('../config/config_pixel_trans_large.py','../config/config_pixel_trans_large_evaluation.py')):
    D = v1_evaluation_protocol(depends_on[1],depends_on[0])    
    actualize(D)    
 
@protocolize()
def pixel_scale_extraction(depends_on = '../config/config_pixel_scale.py'):
    v1_feature_extraction_protocol(depends_on,write=True)

@protocolize()
def pixel_scale_evaluation(depends_on = ('../config/config_pixel_scale.py','../config/config_pixel_scale_evaluation.py')):
    D = v1_evaluation_protocol(depends_on[1],depends_on[0])    
    actualize(D) 
 
@protocolize()
def pixel_rot_extraction(depends_on = '../config/config_pixel_rot.py'):
    v1_feature_extraction_protocol(depends_on,write=True)

@protocolize()
def pixel_rot_evaluation(depends_on = ('../config/config_pixel_rot.py','../config/config_pixel_rot_evaluation.py')):
    D = v1_evaluation_protocol(depends_on[1],depends_on[0])    
    actualize(D)

@protocolize()
def pixel_rot_scale_extraction(depends_on = '../config/config_pixel_rot_scale.py'):
    v1_feature_extraction_protocol(depends_on,write=True)

@protocolize()
def pixel_rot_scale_evaluation(depends_on = ('../config/config_pixel_rot_scale.py','../config/config_pixel_rot_scale_evaluation.py')):
    D = v1_evaluation_protocol(depends_on[1],depends_on[0])    
    actualize(D)


@protocolize()
def pixel_rot_no_preproc_extraction(depends_on = '../config/config_pixel_rot_no_preproc.py'):
    v1_feature_extraction_protocol(depends_on,write=True)

@protocolize()
def pixel_rot_no_preproc_evaluation(depends_on = ('../config/config_pixel_rot_no_preproc.py','../config/config_pixel_rot_no_preproc_evaluation.py')):
    D = v1_evaluation_protocol(depends_on[1],depends_on[0])    
    actualize(D)

@protocolize()
def pixel_rot_no_preproc2_extraction(depends_on = '../config/config_pixel_rot_no_preproc2.py'):
    v1_feature_extraction_protocol(depends_on,write=True)

@protocolize()
def pixel_rot_no_preproc2_evaluation(depends_on = ('../config/config_pixel_rot_no_preproc2.py','../config/config_pixel_rot_no_preproc_evaluation.py')):
    D = v1_evaluation_protocol(depends_on[1],depends_on[0])    
    actualize(D)

@protocolize()
def cairofilters_sq_vs_rect_extraction(depends_on = '../config/config_cairofilters_sq_vs_rect.py'):
    v1_feature_extraction_protocol(depends_on,write=True)



@protocolize()
def cairofilters_sq_vs_rect_trans_evaluation(depends_on = ('../config/config_cairofilters_sq_vs_rect.py','../config/config_cairofilters_sq_vs_rect_trans_evaluation.py')):
    D = v1_evaluation_protocol(depends_on[1],depends_on[0])    
    actualize(D)

@protocolize()
def cairofilters_sq_vs_rect_norm_extraction(depends_on = '../config/config_cairofilters_sq_vs_rect_norm.py'):
    v1_feature_extraction_protocol(depends_on,write=True)

@protocolize()
def cairofilters_sq_vs_rect_norm_trans_evaluation(depends_on = ('../config/config_cairofilters_sq_vs_rect_norm.py','../config/config_cairofilters_sq_vs_rect_norm_trans_evaluation.py')):
    D = v1_evaluation_protocol(depends_on[1],depends_on[0])    
    actualize(D)



def v1_evaluation_protocol(task_config_path,feature_config_path,use_cpu=False):
    oplist = do_initialization(v1_initialize,args = (feature_config_path,use_cpu))
    feature_creates = tuple(oplist[-1]['outcertpaths'])
    hash = get_config_string(oplist[-1]['out_args'])
    
    feature_config = get_config(feature_config_path)
    task_config = get_config(task_config_path)
    
    D = []
    for task in task_config['train_test']:
        c = (feature_config,task)       
        newhash = get_config_string(c)
        outfile = '../.performance_certificates/' + newhash
        op = ('svm_evaluation_' + newhash,train_test_loop,(outfile,feature_creates,task,feature_config_path,hash))
        D.append(op)

    return D    


import pymongo as pm
import gridfs    
@activate(lambda x : x[1],lambda x : x[0])        
def train_test_loop(outfile,extract_creates,task_config,feature_config_path,hash):

    feature_config = get_config(feature_config_path)
        
    base_query = SON([('__config_hash__',hash)])
    
    image_params = SON([('image',feature_config['image'])])
    models_params = feature_config['models']

    ntrain = task_config['ntrain']
    ntest = task_config['ntest']
    ntrain_pos = task_config.get('ntrain_pos')
    N = task_config.get('N',10)
    query = task_config['query']  
    base_query.update(reach_in('config',task_config.get('universe',SON([]))))
 
    print('\n')
    print('BASE',base_query)
    print('\n')
    
    conn = pm.Connection(document_class=SON)
    db = conn['v1']
    fs = gridfs.GridFS(db, collection = 'model_performance')
    
    cquery = reach_in('config',query)
    for m in models_params:
        base_query_copy = base_query.copy()
        base_query_copy.update(reach_in('config.model',m))
        splitdata, results = train_test(cquery,'v1','features',ntrain,ntest,ntrain_pos=ntrain_pos,N=N,universe=base_query_copy)
        
        splitpickle = cPickle.dumps(splitdata)
        
        data = SON([('feature_config_path',feature_config_path),
                    ('model',m),
                    ('task',son_escape(task_config)),
                    ('image__aggregate__',son_escape(feature_config['image']))])
        filename = get_filename(data)
        data.update(results)
        data['filename'] = filename
        

        fs.put(splitpickle,**data)
        
    createCertificateDict(outfile,{'task_config':task_config,'feature_config':feature_config,'feature_config_path':feature_config_path})    

    
#=-=-=-=-=-=-=-=-=
#learning tests
#=-=-=-=-=-=-=-=-=

from v1like_extract import render_image, learn_filterbank

def test_learning_initialize(config_path):
    config = get_config(config_path)
    image_params = SON([('image',config['image'])])
    model_params = SON([('model',config['model'])])
    
    return [('generate_images', render_image, {'args':(image_params,)}),                         
            ('generate_models', learn_filterbank, {'params':model_params})]


def test_learn_protocol(config_path):
    D = DBAdd(test_learning_initialize,args = (config_path,))
    actualize(D)

@protocolize()
def test_learn1(depends_on = '../config/config_test_learning.py'):
    test_learn_protocol(depends_on)
    
        
        