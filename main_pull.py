#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import cPickle
import hashlib

import scipy as sp
import pymongo as pm
import gridfs
from bson import SON
import bson

from starflow.protocols import protocolize, actualize
from starflow.utils import activate

import v1like_extract as v1e
import v1like_funcs as v1f
import traintest
from v1like_extract import get_config
import svm

from dbutils import get_config_string, get_filename, reach_in, DBAdd, createCertificateDict, son_escape, do_initialization, get_most_recent_files


try:
    import v1_pyfft
except:
    GPU_SUPPORT = False
else:
    GPU_SUPPORT = True

@protocolize()
def pull_gridded_gabors_sq_vs_rect_test(depends_on = '../config/config_pull_gridded_gabors_sq_vs_rect_test.py'):
    """basic test of pull protocol with gabor filters, low density transformation image set of squares versus rectangles
    Result: gabor filters do great"""
    D = v1_pull_protocol(depends_on,)
    actualize(D)


@protocolize()
def pull_gridded_gabors_sq_vs_rect(depends_on = '../config/config_pull_gridded_gabors_sq_vs_rect_evaluation.py'):
    """test of standard 96-filter gridded gabor filterbank on high density transformations set of squares versus rectangles
    Result: gabor filters do great"""
    D = v1_pull_protocol(depends_on)
    actualize(D)
    
    
@protocolize()
def pull_gridded_gabors_sq_vs_rect_smallfilters(depends_on = '../config/config_pull_gridded_gabors_sq_vs_rect_smallfilters_evaluation.py'):
    """test of 48-filter gridded gabor filterbank on high density transformations set of squares versus rectangles
    Result: still great"""
    D = v1_pull_protocol(depends_on)
    actualize(D)    
    
    
@protocolize()
def pull_gridded_gabors_sq_vs_rect_verysmallfilters(depends_on = '../config/config_pull_gridded_gabors_sq_vs_rect_verysmallfilters_evaluation.py'):
    """test of 9-filter gridded gabor filterbank on high density transformations set of squares versus rectangles
    Result: STILL great"""
    D = v1_pull_protocol(depends_on)
    actualize(D)        
    
@protocolize()
def pull_gridded_gabors_sq_vs_rect_extremelysmallfilters(depends_on = '../config/config_pull_gridded_gabors_sq_vs_rect_extremelysmallfilters_evaluation.py'):
    """test of 4-filter gridded gabor filterbank on high density transformations set of squares versus rectangles
    Result: STILL quite good"""
    D = v1_pull_protocol(depends_on)
    actualize(D)         
    
@protocolize()
def pull_gridded_gabors_sq_vs_rect_veryveryfewfilters(depends_on = '../config/config_pull_gridded_gabors_sq_vs_rect_veryveryfewfilters_evaluation.py'):
    """test of two-filter gridded gabor filterbank on high density transformations set of squares versus rectangles
    Result: STILL pretty good"""
    D = v1_pull_protocol(depends_on)
    actualize(D)  
    
    
@protocolize()
def pull_gridded_gabors_sq_vs_rect_onefilter(depends_on = '../config/config_pull_gridded_gabors_sq_vs_rect_onefilter_evaluation.py'):
    """test of 1-filter 'gridded' gabor filterbank on high density transformations set of squares versus rectangles
    Result: not great""" 
    D = v1_pull_protocol(depends_on)
    actualize(D)     
    
@protocolize()
def pull_gridded_gabors_sq_vs_rect_varioustwofilters(depends_on = '../config/config_pull_gridded_gabors_sq_vs_rect_varioustwofilters_evaluation.py'):
    """test of two-filter orthogonal orientation gridded gabor filterbanks of various kshapes and frequencies on high density transformations set of
    squares versus rectangles.
    RESULT: most do quite well.   Smaller kshape and higher frequencies in this test do better. """
    D = v1_pull_protocol(depends_on)
    actualize(D)      

@protocolize()
def pull_gridded_gabors_sq_vs_rect_various_twofrequency_filterbanks(depends_on = '../config/config_pull_gridded_gabors_sq_vs_rect_various_twofrequency_filterbanks.py'):
    """test of two-frequency gridded gabor filterbanks of various kshapes and fixed single orientation on high density transformations set of
    squares versus rectangles.
    RESULT: Varied """
    D = v1_pull_protocol(depends_on)
    actualize(D)      
        

@protocolize()
def pull_random_gabors_sq_vs_rect_onefilter_screen(depends_on = '../config/config_pull_random_gabors_sq_vs_rect_onefilter_screen.py'):
    """screening 10 random one-filter gabors on high density transformations set of squares versus rectangles
    Result: all suck """
    D = v1_pull_protocol(depends_on)
    actualize(D)
    
    
@protocolize()
def pull_random_gabors_sq_vs_rect_twofilter_screen(depends_on = '../config/config_pull_random_gabors_sq_vs_rect_twofilter_screen.py'):
    """screening 10 randomly-oriented two-gabors filterbanks with fixed frequency and pahse on high density transformations set of squares versus rectangles
    Result: all suck""" 
    D = v1_pull_protocol(depends_on)
    actualize(D)    
    
@protocolize()
def pull_gabor_sq_vs_rect_twofilter_pump_training(depends_on = '../config/config_pull_gabor_sq_vs_rect_twofilter_pump_training.py'):
    """taking one of the best (but still bad) performing random two-filter gabors and pumping up traning examples on high density transformations 
    set of squares versus rectangles
    Result: still bad"""
    D = v1_pull_protocol(depends_on)
    actualize(D)    
    
@protocolize()
def pull_cairofilters_sq_vs_rect_various_activations(depends_on = '../config/config_pull_cairofilters_sq_vs_rect_various_activations.py'):
    """tuning activation threshold on handcrafted cairo filters on high density transformations set of squares versus rectangles"""
    D = v1_pull_protocol(depends_on)
    actualize(D)  
    
@protocolize()
def pull_cairofilters_sq_vs_rect_various_activations_finetuning(depends_on = '../config/config_pull_cairofilters_sq_vs_rect_various_activations_finetuning.py'):
    """finetuning activation threshold on handcrafted cairo filters on high density transformations set of squares versus rectangles"""
    D = v1_pull_protocol(depends_on)
    actualize(D)      
    
@protocolize()
def pull_cairofilters_sq_vs_rect_various_activations_finefinetuning(depends_on = '../config/config_pull_cairofilters_sq_vs_rect_various_activations_finefinetuning.py'):
    """finefinetuning activation threshold on handcrafted cairo filters on high density transformations set of squares versus rectangles"""
    D = v1_pull_protocol(depends_on)
    actualize(D)   
    
    
    
@protocolize()
def pull_activation_tuned_cairofilters_sq_vs_rect(depends_on = '../config/config_pull_activation_tuned_cairofilters_sq_vs_rect.py'):
    """pumped-up trainining curve evaluation on  handcrafted cairo filters with optimized activation valued from finefinetuning on high density transformations set of squares versus rectangles
       result: you can pump up performance into > 95%. 
    """
    
    D = v1_pull_protocol(depends_on)
    actualize(D)          
    
 
def v1_pull_evaluation_protocol(im_config_path,task_config_path,use_cpu = False,write=False):
    
    oplist = do_initialization(pull_initialize,args = (im_config_path,))    
    image_certificate = oplist[0]['outcertpaths'][0]
    model_certificate = oplist[1]['outcertpaths'][0]
    
    if use_cpu or not GPU_SUPPORT:    
        convolve_func = v1f.v1like_filter_numpy
    else:
        convolve_func = v1f.v1like_filter_pyfft

    config = get_config(task_config_path)
    task_config = config.pop('train_test')
    D = []
    for task in task_config:
        c = (config,task)       
        newhash = get_config_string(c)
        outfile = '../.performance_certificates/' + newhash
        op = ('svm_evaluation_' + newhash,train_test_pull,(outfile,task,image_certificate,model_certificate,convolve_func))
        D.append(op)

    if write:
        actualize(D)
    return D


def v1_pull_protocol(config_path,use_cpu = False,write=False):

    D = DBAdd(pull_initialize,args = (config_path,))
    
    oplist = do_initialization(pull_initialize,args = (config_path,))    
    image_certificate = oplist[0]['outcertpaths'][0]
    model_certificate = oplist[1]['outcertpaths'][0]
    
    if use_cpu or not GPU_SUPPORT:    
        convolve_func = v1f.v1like_filter_numpy
    else:
        convolve_func = v1f.v1like_filter_pyfft

    config = get_config(config_path)
    task_config = config.pop('train_test')
    
    for task in task_config:
        c = (config,task)       
        newhash = get_config_string(c)
        outfile = '../.performance_certificates/' + newhash
        op = ('svm_evaluation_' + newhash,train_test_pull,(outfile,task,image_certificate,model_certificate,convolve_func))
        D.append(op)

    if write:
        actualize(D)
    return D


def pull_initialize(config_path):
    config = get_config(config_path)    
    image_params = SON([('image',config['image'])])
    models_params = config['models']
    for model_params in models_params:
        if model_params['filter']['model_name'] in ['really_random','random_gabor']:
            #model_params['id'] = v1e.random_id()
            pass
    
    return [{'step':'generate_images','func':v1e.render_image, 'params':(image_params,)},                         
            {'step':'generate_models', 'func':v1e.get_filterbank,'params':(models_params,)},            
           ]

        
@activate(lambda x : (x[2],x[3]),lambda x : x[0])
def train_test_pull(outfile,task,image_certificate_file,model_certificate_file,convolve_func):

    conn = pm.Connection(document_class=bson.SON)
    db = conn['v1']
    
    perf_fs = gridfs.GridFS(db,'performance')
    
    model_coll = db['models.files']
    model_fs = gridfs.GridFS(db,'models')
    image_coll = db['raw_images.files']
    image_fs = gridfs.GridFS(db,'raw_images')
    
    image_certdict = cPickle.load(open(image_certificate_file))
    model_certdict = cPickle.load(open(model_certificate_file))
    print('using image certificate', image_certificate_file)
    print('using model certificate', model_certificate_file)    
    image_hash = image_certdict['run_hash']
    model_hash = model_certdict['run_hash']
    image_args = image_certdict['out_args']
    
    stats = ['test_accuracy','ap','auc','mean_ap','mean_auc','train_accuracy']    
    classifier_kwargs = task.get('classifier_kwargs',{})
    
    model_configs = get_most_recent_files(model_coll,{'__run_hash__':model_hash})
    
    if convolve_func == v1f.v1like_filter_pyfft:
        v1_pyfft.setup_pyfft()
    for m in model_configs:
        split_results = []
        splits = generate_splits(task,image_hash) 
        for (ind,split) in enumerate(splits):
            print ('split', ind)
            train_data = split['train_data']
            test_data = split['test_data']
            
            train_filenames = [t['filename'] for t in train_data]
            test_filenames = [t['filename'] for t in test_data]
            assert set(train_filenames).intersection(test_filenames) == set([])
            
            train_features = sp.row_stack([extract_features(im, image_fs, m, model_fs, convolve_func) for im in train_data])
            test_features = sp.row_stack([extract_features(im, image_fs, m, model_fs, convolve_func) for im in test_data])
            train_labels = split['train_labels']
            test_labels = split['test_labels']

            res = svm.classify(train_features,train_labels,test_features,test_labels,**classifier_kwargs)
 
            split_results.append(res)
    
        model_results = SON([])
        for stat in stats:
            if stat in split_results[0] and split_results[0][stat] != None:
                model_results[stat] = sp.array([split_result[stat] for split_result in split_results]).mean()           

        out_record = SON([('model',m['config']['model']),
                       ('model_filename',m['filename']),
                       ('task',son_escape(task)),
                       ('images',son_escape(image_args)),
                       ('images_hash',image_hash),
                       ('models_hash',model_hash)
                     ])   
        filename = get_filename(out_record)
        out_record['filename'] = filename
        out_record.update(model_results)
        out_data = cPickle.dumps(SON([('split_results',split_results),('splits',splits)]))
        
        perf_fs.put(out_data,**out_record)
 
    if convolve_func == v1f.v1like_filter_pyfft:
        v1_pyfft.cleanup_pyfft() 
      
    createCertificateDict(outfile,{'image_file':image_certificate_file,'models_file':model_certificate_file})
    
FEATURE_CACHE = {}

def get_from_cache(obj,cache):
    hash = hashlib.sha1(repr(obj)).hexdigest()
    if hash in cache:
        return cache[hash]
        
def put_in_cache(obj,value,cache):
    hash = hashlib.sha1(repr(obj)).hexdigest()
    cache[hash] = value

def extract_features(image_config, image_fs, model_config, model_fs,convolve_func):

    cached_val = get_from_cache((image_config,model_config),FEATURE_CACHE)
    if cached_val is not None:
        output = cached_val
    else:
        print('extracting', image_config, model_config)
        
        image_fh = image_fs.get_version(image_config['filename'])
        filter_fh = model_fs.get_version(model_config['filename'])
        
        m_config = model_config['config']['model']
        conv_mode = m_config['conv_mode']
        
        #preprocessing
        array = v1e.image2array(m_config ,image_fh)
        
        preprocessed,orig_imga = v1e.preprocess(array,m_config )
            
        #input normalization
        norm_in = v1e.norm(preprocessed,conv_mode,m_config.get('normin'))
        
        #filtering
        filtered = v1e.convolve(norm_in, filter_fh, m_config , convolve_func)
        
        #nonlinear activation
        activ = v1e.activate(filtered,m_config.get('activ'))
        
        #output normalization
        norm_out = v1e.norm(activ,conv_mode,m_config.get('normout'))
        #pooling
        pooled = v1e.pool(norm_out,conv_mode,m_config.get('pool'))
            
        #postprocessing
        fvector_l = v1e.postprocess(norm_in,filtered,activ,norm_out,pooled,orig_imga,m_config.get('featsel'))
        
        output = sp.concatenate(fvector_l).ravel()
        put_in_cache((image_config,model_config),output,FEATURE_CACHE)
    
    return output
    
    
def generate_splits(task_config,image_hash):
    
    base_query = SON([('__run_hash__',image_hash)])
    ntrain = task_config['ntrain']
    ntest = task_config['ntest']
    ntrain_pos = task_config.get('ntrain_pos')
    N = task_config.get('N',10)
    query = task_config['query']  
    base_query.update(reach_in('config',task_config.get('universe',SON([]))))    
    cquery = reach_in('config',query)
    
    print ('query',cquery)
    print ('universe',base_query)
    
    return [traintest.generate_split2('v1','raw_images',cquery,ntrain,ntest,ntrain_pos=ntrain_pos,universe=base_query) for ind in range(N)]
        