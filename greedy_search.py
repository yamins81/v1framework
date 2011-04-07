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

import config_modifiers
import v1like_extract as v1e
import v1like_funcs as v1f
import traintest
from v1like_extract import get_config
import svm

from dbutils import get_config_string, get_filename, reach_in, DBAdd, createCertificateDict, son_escape, do_initialization, get_most_recent_files, hgetattr, hsetattr

from main_pull import generate_splits, pull_initialize, get_from_cache, put_in_cache

FEATURE_CACHE = {}

try:
    import v1_pyfft
except:
    GPU_SUPPORT = False
else:
    GPU_SUPPORT = True


###TODO: Make search process more sophisticated, right now it sucks.  

@protocolize()
def pull_gridded_gabors_sq_vs_rect_onefilter(depends_on = '../config/config_greedy_optimization_onegabor_filterbank_sq_vs_rect.py'):
    """
    Greedy search for better single-gabor sq vs rectangle.   
    RESULT:  it's all bad  
    """ 
    D = v1_greedy_optimization_protocol(depends_on)
    actualize(D)     
    
@protocolize()
def pull_gridded_gabors_sq_vs_rect_onefilter_coarser(depends_on = '../config/config_greedy_optimization_onegabor_filterbank_sq_vs_rect_coarser.py'):
    """
    Greedy search for better single-gabor sq vs rectangle, with coarser step in search
    RESULT:  it's all bad 
    """ 
    D = v1_greedy_optimization_protocol(depends_on)
    actualize(D)     


@protocolize()
def optimize_gridded_gabors_sq_vs_rect_twofilter(depends_on = '../config/config_greedy_optimization_twogabor_filterbank_sq_vs_rect.py'):
    """
    Greedy search for best two-orthogonal-gabor filterbank sq vs rectangle
    RESULT:  you get to v. high performance quickly (and this could really benefit from better search procedure to optimize further)
    """ 
    D = v1_greedy_optimization_protocol(depends_on)
    actualize(D)     


def v1_greedy_optimization_protocol(config_path,use_cpu = False,write=False):

    D = DBAdd(image_initialize,args = (config_path,))
    
    oplist = do_initialization(image_initialize,args = (config_path,))    
    image_certificate = oplist[0]['outcertpaths'][0]
    
    if use_cpu or not GPU_SUPPORT:    
        convolve_func = v1f.v1like_filter_numpy
    else:
        convolve_func = v1f.v1like_filter_pyfft

    config = get_config(config_path)
    
    task = config['evaluation_task']
    initial_model = config['model']
    
    modifier_args = config['modifier_args']
    modifier_class = config.get('modifier')
    rep_limit = config.get('rep_limit')
    
    if modifier_class is None:
        modifier = config_modifiers.BaseModifier(modifier_args)
    else:
        modifier = modifier_class(modifier_args)
              
    newhash = get_config_string(config)
    outfile = '../.optimization_certificates/' + newhash
    op = ('optimization_' + newhash,greedy_optimization,(outfile,task,image_certificate,initial_model,convolve_func,rep_limit,modifier_args,modifier))
    D.append(op)

    if write:
        actualize(D)
    return D


def image_initialize(config_path):
    config = get_config(config_path)    
    image_params = SON([('image',config['image'])])
    
    return [{'step':'generate_images','func':v1e.render_image, 'params':(image_params,)},                                    
           ]

import filter_generation

@activate(lambda x : x[2],lambda x : x[0])
def greedy_optimization(outfile,task,image_certificate_file,initial_model,convolve_func,rep_limit, modifier_args,modifier):

    conn = pm.Connection(document_class=bson.SON)
    db = conn['v1']
    
    opt_fs = gridfs.GridFS(db,'optimized_performance')
    
    image_coll = db['raw_images.files']
    image_fs = gridfs.GridFS(db,'raw_images')
    
    image_certdict = cPickle.load(open(image_certificate_file))
    print('using image certificate', image_certificate_file)
    
    image_hash = image_certdict['run_hash']
    image_args = image_certdict['out_args']

    if convolve_func == v1f.v1like_filter_pyfft:
        v1_pyfft.setup_pyfft()
    
  
    filterbanks = []
    perfs = []
    model_configs = []
    center_config = initial_model
    
    i = 0
    improving = True
    
    
    while ((i < rep_limit) or rep_limit is None):
        i += 1
        print('Round', i)
        next_configs = [m for m in get_consistent_deltas(center_config,modifier) if m not in model_configs]

        if next_configs:
            next_results = [get_performance(task,image_hash,image_fs,m,convolve_func) for m in next_configs]
            next_perfs = [x[0] for x in next_results]
            next_filterbanks = [x[1] for x in next_results]
            next_perf_ac_max = np.array([x['test_accuracy'] for x in next_perfs]).max()
            perf_ac_max = max([x['test_accuracy'] for x in perfs]) if perfs else 0
            if next_perf_ac_max > perf_ac_max:
                next_perf_ac_argmax = np.array([x['test_accuracy'] for x in next_perfs]).argmax()
                center_config = next_configs[next_perf_ac_argmax]  
                print('\n\n')
                print('new best performance is', next_perf_ac_max, 'from model', center_config)
                print('\n\n')
                perfs.extend(next_perfs)  
                model_configs.extend(next_configs)
                filterbanks.extend(next_filterbanks)
            else:
                print('Breaking because no further optimization could be done.  Best existing performance was', perf_ac_max, 'while best next performance was', next_perf_ac_max)
                break
            
        else:
            print('Breaking because no next configs')
            break
        

    perfargmax = np.array([p['test_accuracy'] for p in perfs]).argmax()
    best_model = model_configs[perfargmax]
    best_performance = perfs[perfargmax]
        
    out_record = SON([('initial_model',initial_model),
                       ('task',son_escape(task)),
                       ('images',son_escape(image_args)),
                       ('images_hash',image_hash),
                       ('modifier_args',son_escape(modifier_args)),
                       ('modifier',modifier.__class__.__module__ + '.' + modifier.__class__.__name__)
                     ])   
    filename = get_filename(out_record)
    out_record['filename'] = filename
    out_record.update(SON([('performances',perfs)]))
    out_record.update(SON([('best_model',best_model)]))
    out_record.update(SON([('best_performance',best_performance)]))
    out_record.update(SON([('num_steps',len(model_configs))]))
    out_record.update(SON([('models',model_configs)]))
    outdata = cPickle.dumps(filterbanks)
        
    opt_fs.put(outdata,**out_record)
     
    if convolve_func == v1f.v1like_filter_pyfft:
        v1_pyfft.cleanup_pyfft() 
      
    createCertificateDict(outfile,{'image_file':image_certificate_file})

import numpy as np
def greedy_modify_config(model_configs,perfs,modifier):
    
    perfvals = [p['test_accuracy'] for p in perfs]
    perfargmax = np.array(perfvals).argmax()
    perfmax = model_configs[perfargmax]

    possible_next_configs = get_consistent_deltas(perfmax,modifier)
    
    perfargmax0 = np.array(perfvals[:perfargmax]).argmax() if perfargmax > 0 else None
    print perfvals, perfargmax, perfargmax0
    if perfargmax0 is not None:
        perfmax0 = model_configs[perfargmax0] 
        possible_next_configs = greedy_order(possible_next_configs,perfmax0,perfmax,modifier)
    
    untried_next_configs = [x for x in possible_next_configs if x not in model_configs]
    
    if untried_next_configs:
        return untried_next_configs[0]
        
def greedy_order(configs,x0,x1,modifier):
    dist_vec = np.array([modifier.get_vector(x0,x1,k) for k in modifier.params])
    dist_vecs = [np.array([modifier.get_vector(x1,y,k) for k in modifier.params]) for y in configs]
    dist_dots = np.array([np.dot(dist_vec,d_vec) for d_vec in dist_vecs])
    
    ordering = dist_dots.argsort()[::-1]
    
    configs = [configs[ind] for ind in ordering]
    
    return configs
    

from copy import deepcopy    
import itertools
def get_consistent_deltas(perfmax,modifier):
    
    possibles = []
    
    LL = [modifier.get_modifications(k,hgetattr(perfmax,k)) for k in modifier.params]

    D = itertools.product(*LL)
    
    for d in D:
        c_copy = deepcopy(perfmax)
        for (ind,k) in enumerate(modifier.params):
            print k, d[ind]
            hsetattr(c_copy,k,d[ind]) 
       
        possibles.append(c_copy)
            
    return possibles
    
    
 

def get_performance(task,image_hash,image_fs,model_config,convolve_func):
    stats = ['test_accuracy','ap','auc','mean_ap','mean_auc','train_accuracy']    
    classifier_kwargs = task.get('classifier_kwargs',{})

    split_results = []  
    splits = generate_splits(task,image_hash) 
    filterbank = filter_generation.get_filterbank(model_config)
    for (ind,split) in enumerate(splits):
        print ('split', ind)
        train_data = split['train_data']
        test_data = split['test_data']
        
        train_filenames = [t['filename'] for t in train_data]
        test_filenames = [t['filename'] for t in test_data]
        assert set(train_filenames).intersection(test_filenames) == set([])
        
        train_features = sp.row_stack([extract_features(im, image_fs, filterbank, model_config, convolve_func) for im in train_data])
        test_features = sp.row_stack([extract_features(im, image_fs, filterbank, model_config, convolve_func) for im in test_data])
        train_labels = split['train_labels']
        test_labels = split['test_labels']

        res = svm.classify(train_features,train_labels,test_features,test_labels,**classifier_kwargs)

        split_results.append(res)

    model_results = SON([])
    for stat in stats:
        if stat in split_results[0] and split_results[0][stat] != None:
            model_results[stat] = sp.array([split_result[stat] for split_result in split_results]).mean()           


    return model_results, filterbank
    

def extract_features(image_config, image_fs, filterbank, model_config, convolve_func):

    cached_val = get_from_cache((image_config,model_config),FEATURE_CACHE)
    if cached_val is not None:
        output = cached_val
    else:
        print('extracting', image_config, model_config)
        
        image_fh = image_fs.get_version(image_config['filename'])
        
        
        m_config = model_config
        conv_mode = m_config['conv_mode']
        
        #preprocessing
        array = v1e.image2array(m_config ,image_fh)
        
        preprocessed,orig_imga = v1e.preprocess(array,m_config )
            
        #input normalization
        norm_in = v1e.norm(preprocessed,conv_mode,m_config.get('normin'))
        
        #filtering
        filtered = v1e.convolve(norm_in, filterbank, m_config , convolve_func)
        
        #nonlinear activation
        activ = v1e.activate(filtered,m_config.get('activ'))
        
        #output normalization
        norm_out = v1e.norm(activ,conv_mode,m_config.get('normout'))
        #pooling
        pooled = v1e.pool(norm_out,conv_mode,m_config.get('pool'))
            
        #postprocessing
        fvector_l = v1e.postprocess(norm_in,filtered,activ,norm_out,pooled,orig_imga,m_config.get('featsel'))
        
        output = sp.concatenate(fvector_l).ravel()
        put_in_cache((image_config,m_config),output,FEATURE_CACHE)
    
    return output
