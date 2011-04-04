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

from dbutils import get_config_string, get_filename, reach_in, DBAdd, createCertificateDict, son_escape, do_initialization, get_most_recent_files

from main_pull import generate_splits, pull_initialize

try:
    import v1_pyfft
except:
    GPU_SUPPORT = False
else:
    GPU_SUPPORT = True


@protocolize()
def pull_gridded_gabors_sq_vs_rect_onefilter(depends_on = '../config/config_greedy_optimization_onegabor_filterbank_sq_vs_rect.py'):
    """
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
    initial_model = config['initial_model']
    
    modifier_args = config['modifier_args']
    modifier_class = config.get('modifier')
    
    if modifier is None:
        modifier = config_modifiers.BaseModifier(modifier_args)
    else:
        modifier = modifier_class(modifier_args)
              
	newhash = get_config_string(config)
	outfile = '../.optimization_certificates/' + newhash
	op = ('optimization_' + newhash,greedy_optimization,(outfile,task,image_certificate,initial_model,convolve_func,modifier_args,modifier))
	D.append(op)

    if write:
        actualize(D)
    return D


def image_initialize(config_path):
    config = get_config(config_path)    
    image_params = SON([('image',config['image'])])
    
    return [{'step':'generate_images','func':v1e.render_image, 'params':(image_params,)},                                    
           ]


@activate(lambda x : x[2],lambda x : x[0])
def greedy_optimization(outfile,task,image_certificate_file,initial_model,convolve_func,modifier_args,modifier):

    conn = pm.Connection(document_class=bson.SON)
    db = conn['v1']
    
    perf_fs = gridfs.GridFS(db,'optimized_performance')
    
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
	
	while model_config:
		model_configs.append(model_config)
		perf,filterbank = get_performance(task,image_hash,image_coll,model_config,convolve_func)
		perfs.append(perf)	
		filterbanks.append(filterbank)
		model_config = greedy_modify_config(model_configs,perfs,modifier)
	
	best_model = model_config[np.array(perfs).argmax()]
	best_performance = perfs.max()
		
	out_record = SON([('initial_model',model_config['config']['model']),
					   ('model_filename',model_config['filename']),
					   ('task',son_escape(task)),
					   ('images',son_escape(image_args)),
					   ('images_hash',image_hash),
					   ('models_hash',model_hash),
					   ('modifier_args',modifier_args),
					   ('modifier',modifier.__class__)
					 ])   
	filename = get_filename(out_record)
	out_record['filename'] = filename
	out_record.update(SON([('performances',perfs)]))
	out_record.update(SON([('best_model',best_model)]))
	out_record.update(SON([('best_performance',best_performance)]))
	out_record.update(SON([('num_steps',len(model_configs))]))
	out_record.update(SON([('models',model_configs)]))
	outdata = cPickle.dumps(filterbanks)
		
	perf_fs.put(outdata,**out_record)
     
    if convolve_func == v1f.v1like_filter_pyfft:
        v1_pyfft.cleanup_pyfft() 
      
    createCertificateDict(outfile,{'image_file':image_certificate_file,'models_file':model_certificate_file})


def greedy_modify_config(model_configs,perfs,modifier):
    
    perfargmax = np.array(perfs).argmax()
    perfmax = model_configs[perfargmax]
    perfmax0 = model_configs[np.array(perfs[:perfargmax]).argmax()] if perfargmax else None
    
    possible_next_configs = get_consistent_deltas(perfmax0,perfmax,modifier)
    
    untried_next_configs = [x for x in possible_next_configs if x not in model_configs[perfargmax:]]
    
    if untried_next_configs:
        return untried_next_configs[0]
        
        
from copy import deepcopy      
def get_consistent_deltas(perfmax0,perfmax,modifier):
    
    possibles = []
    
    D = itertools.product([modifier.get_modifications(k,hgetattr(perfmax,k)) for k in modifier.params])
    
    for d in D:
        c_copy = deepcopy(perfmax)
        for (ind,k) in enumerate(modification_params.keys()):
            hsetattr(c_copy,k,d[ind]) 
       
        if c_copy != perfmax0 and c_copy != perfmax:
            possibles.append(c_copy)
            
    return possibles
    
    
def hsetattr(d,k,v):
    kl = k.split('.')
    h_do_setattr(d,kl,v)
    
def h_do_setattr(d,kl,v):
    if len(kl) == 1:
        d[kl[0]] = v
    else:
        h_do_setattr(d[kl[0]],kl[1:],v)
        
def hgetattr(d,k,v):
    kl = k.split('.')
    h_do_getattr(d,kl,v)
    
def h_do_getattr(d,kl,v):
    if len(kl) == 1:
        return d[kl[0]]
    else:
        h_do_getattr(d[kl[0]],kl[1:],v) 
 

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
        
        
        m_config = model_config['config']['model']
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
        put_in_cache((image_config,model_config),output,FEATURE_CACHE)
    
    return output
