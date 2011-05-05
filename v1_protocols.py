from sge_utils import qsub

import sys
import cPickle
import hashlib
import os
import random

import scipy as sp
import numpy as np
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
import rendering
import filter_generation

from dbutils import get_config_string, get_filename, reach_in, createCertificateDict, son_escape, get_most_recent_files

DB_NAME = 'v1-test'

try:
    import pycuda.driver as cuda
    cuda.init()
except:
    GPU_SUPPORT = False
else:
    import v1_pyfft
    GPU_SUPPORT = True

def remove_existing(coll,fs, hash):
    existing = coll.find({'__hash__':hash})
    for e in existing:
        fs.delete(e['_id'])

def image_protocol(config_path,write = False):

    config = get_config(config_path)

    image_hash = get_config_string(config['images'])
    image_certificate = '../.image_certificates/' + image_hash

    parallel = config['images'].get('generate_parallel',False)
    if  not parallel:
        D = [('generate_images',generate_images,(image_certificate,image_hash,config))]
    else:
        D = [('generate_images',generate_images_parallel,(image_certificate,image_hash,config))]
    
    if write:
        actualize(D)
    return D


@activate(lambda x : (), lambda x : x[0])    
def generate_images(outfile,im_hash,config_gen):

    conn = pm.Connection(document_class = SON)
    db = conn[DB_NAME]
    im_coll = db['images.files']
    im_fs = gridfs.GridFS(db,'images')
    
    remove_existing(im_coll,im_fs,im_hash)
    
    X = rendering.config_gen(config_gen)
    
    for (i,x) in enumerate(X):
        x['image']['generator'] = config_gen['images']['generator']
        if (i/100)*100 == i:
            print(i,x) 
        
        image_string = rendering.render_image(x['image']) 
        y = SON([('config',x)])
        filename = get_filename(x)
        y['filename'] = filename
        y['__hash__'] = im_hash
        im_fs.put(image_string,**y)
        
    createCertificateDict(outfile,{'image_hash':im_hash,'args':config_gen})

def generate_and_insert_single_image(x,im_hash):

    conn = pm.Connection(document_class = SON)
    db = conn[DB_NAME]
    im_coll = db['images.files']
    im_fs = gridfs.GridFS(db,'images')
    
    image_string = rendering.render_image(x['image']) 
    y = SON([('config',x)])
    filename = get_filename(x)
    y['filename'] = filename
    y['__hash__'] = im_hash
    im_fs.put(image_string,**y)


@activate(lambda x : (), lambda x : x[0])    
def generate_images_parallel(outfile,im_hash,config_gen):

    conn = pm.Connection(document_class = SON)
    db = conn[DB_NAME]
    im_coll = db['images.files']
    im_fs = gridfs.GridFS(db,'images')
    
    remove_existing(im_coll,im_fs,im_hash)
    
    X = rendering.config_gen(config_gen)
       
    jobids = []
    for (i,x) in enumerate(X):
        jobid = qsub(generate_and_insert_single_image,(x,im_hash),queueName='images')  
        jobids.append(jobid)
        
    createCertificateDict(outfile,{'image_colname':colname,'args':config_gen})

    return {'child_jobs':jobids}
    
   
def model_config_generator(config): 
    models = config['models']
    return [SON([('model',m)]) for m in models]   
    
    
def model_protocol(config_path,write = False):

    config = get_config(config_path)

    model_hash = get_config_string(config['models'])
    model_certificate = '../.model_certificates/' + model_hash

    D = [('generate_models',generate_models,(model_certificate,model_hash,config))]
    
    if write:
        actualize(D)
    return D


@activate(lambda x : (), lambda x : x[0])    
def generate_models(outfile,m_hash,config_gen):

    conn = pm.Connection(document_class = SON)
    db = conn[DB_NAME]
    m_coll = db['models.files']
    m_fs = gridfs.GridFS(db,'models')
    
    remove_existing(m_coll,m_fs,m_hash)
    
    M = model_config_generator(config_gen)       
    
    for (i,m) in enumerate(M):
        filterbank = filter_generation.get_filterbank(m['model']) 
        filterbank_string = cPickle.dumps(filterbank)
        if (i/100)*100 == i:
            print(i,m) 
        
        y = SON([('config',m)])
        filename = get_filename(m)
        y['filename'] = filename
        y['__hash__'] = m_hash
        m_fs.put(filterbank_string,**y)
        
    createCertificateDict(outfile,{'model_hash':m_hash,'args':config_gen})
    

#goes through everything in relevant collections and loads and does extraction into named new collection.   either in parallel or not
#in parallel, it splits things up into batches of some size 

def extract_features_protocol(image_config_path,model_config_path,feature_config_path,convolve_func_name = 'numpy', write = False):

    model_config_gen = get_config(model_config_path)
    model_hash = get_config_string(model_config_gen['models'])
    model_certificate = '../.model_certificates/' + model_hash
    
    image_config_gen = get_config(image_config_path)
    image_hash = get_config_string(image_config_gen['images'])
    image_certificate = '../.image_certificates/' + image_hash
    
    overall_config_gen = SON([('models',model_config_gen['models']),('images',image_config_gen['images'])])
    feature_hash = get_config_string(overall_config_gen)
    feature_certificate = '../.feature_certificates/' + feature_hash
    
    feature_config = get_config(feature_config_path)
    feature_config.update(overall_config_gen)
    parallel = feature_config.get('feature_extraction',{}).get('generate_parallel',False)
    batch_size = feature_config.get('feature_extraction',{}).get('batch_size',1000)
 
  
    if not parallel:
        D = [('extract_features',extract_features,(feature_certificate,image_certificate,model_certificate,feature_config,feature_hash,convolve_func_name))]
    else:
        D = [('extract_features',extract_features_parallel,(feature_certificate,image_certificate,model_certificate,feature_config,feature_hash,convolve_func_name,batch_size))]
    
    if write:
        actualize(D)
    return D


@activate(lambda x : (x[1],x[2]), lambda x : x[0])    
def extract_features(feature_certificate,image_certificate,model_certificate,feature_config,feature_hash,convolve_func_name):

    image_certdict = cPickle.load(open(image_certificate))
    image_hash = image_certdict['image_hash']
    image_args = image_certdict['args']

    model_certdict = cPickle.load(open(model_certificate))
    model_hash = model_certdict['model_hash']
    model_args = model_certdict['args']

    conn = pm.Connection(document_class = SON)
    db = conn[DB_NAME]
    f_coll = db['features.files']
    f_fs = gridfs.GridFS(db,'features')
    
    remove_existing(f_coll,f_fs,feature_hash)
        
    extract_features_core(image_certificate,model_certificate,feature_hash,image_hash,model_hash,convolve_func_name)
     
    createCertificateDict(feature_certificate,{'feature_hash':feature_hash,
                                               'image_hash':image_hash,
                                               'model_hash':model_hash,
                                               'args':feature_config,})
   
 
def get_device_id():
    num_gpus = get_num_gpus()
    return os.environ.get('JOB_ID',random.randint(0,num_gpus-1)) % num_gpus
    
def get_num_gpus():
    num = 0
    while True:
        try:
            cuda.Device(num)
        except:
            break
        else:
            num += 1
    return num
    
def extract_features_core(image_certificate,model_certificate,feature_hash,image_hash,model_hash,convolve_func_name,im_query=None,m_query=None,im_skip=0,im_limit=0,m_skip=0,m_limit=0):
    if im_query is None:
        im_query = {}
    if m_query is None:
        m_query = {}
        
    im_query['__hash__'] = image_hash
    m_query['__hash__'] = model_hash

    conn = pm.Connection(document_class = SON)
    db = conn[DB_NAME]
    
    image_col = db['images.files'] 
    model_col = db['models.files'] 
    
    image_fs = gridfs.GridFS(db,'images')
    model_fs = gridfs.GridFS(db,'models')
    
    feature_fs = gridfs.GridFS(db,'features')
    
    if convolve_func_name == 'numpy':
        convolve_func = v1f.v1like_filter_numpy
    elif convolve_func_name == 'pyfft':
        convolve_func = v1f.v1like_filter_pyfft
    else:
        raise ValueError, 'convolve func name not recognized'
        
    if convolve_func == v1f.v1like_filter_pyfft:
        device_id = get_device_id() 
        v1_pyfft.setup_pyfft(device_id = device_id)         

    for image_config in get_most_recent_files(image_col,im_query,skip=im_skip,limit=im_limit):
        for model_config in get_most_recent_files(model_col,m_query,skip=m_skip,limit=m_limit):              
            print('extracting',image_config,model_config)
            features = compute_features(image_config, image_fs, model_config, model_fs,convolve_func)
            features_string = cPickle.dumps(features)
            y = SON([('config',SON([('model',model_config['config']['model']),('image',image_config['config']['image'])]))])
            filename = get_filename(y['config'])
            y['filename'] = filename
            y['__hash__'] = feature_hash
            feature_fs.put(features_string,**y)
            
    if convolve_func == v1f.v1like_filter_pyfft:
        v1_pyfft.cleanup_pyfft()            
            
        
def get_feature_batches(im_hash,m_hash,im_col,m_col,batch_size):
    im_count = im_col.count({'__hash__':im_hash})
    m_count = m_col.count({'__hash__':m_hash})
     
    im_batches = [(batch_size*i,batch_size*(i+1)) for i in range(max(im_count/batch_size,1))]
    m_batches = [(j,j+1) for j in range(m_count)]
    
    return [(imb[0],imb[1],mb[0],mb[1]) for imb in im_batches for mb in m_batches]
    

        
@activate(lambda x : (x[1],x[2]), lambda x : x[0])    
def extract_features_parallel(feature_certificate,image_certificate,model_certificate,feature_config,feature_hash,convolve_func_name,batch_size):
    conn = pm.Connection(document_class = SON)
    db = conn[DB_NAME]
    
    image_certdict = cPickle.load(open(image_certificate))
    image_hash = image_certdict['image_hash']
    image_args = image_certdict['args']

    model_certdict = cPickle.load(open(model_certificate))
    model_hash = model_certdict['model_hash']
    model_args = model_certdict['args']

    conn = pm.Connection(document_class = SON)
    db = conn[DB_NAME]
    f_coll = db['features.files']
    f_fs = gridfs.GridFS(db,'features')
    
    remove_existing(f_coll,f_fs,feature_hash)
    
    limits = get_feature_batches(image_hash,model_hash,db['images.files'],db['models.files'],batch_size)
    
    jobids = []

    for (ind,limit) in enumerate(limits):
        im_from,im_to,m_from,m_to = limit
        jobid = qsub(extract_features_core,[(image_certificate,
                                             model_certificate,
                                             feature_hash,
                                             image_hash,
                                             model_hash,
                                             convolve_func_name),
                                             {'im_skip':im_from,
                                              'im_limit':im_to-im_from,
                                              'm_skip':m_from,
                                              'm_limit':m_to-m_from}],
                                            queueName='feature_extraction')
        jobids.append(jobid)

    createCertificateDict(feature_certificate,{'feature_colname':feature_hash,
                                               'image_colname':image_hash,
                                               'model_colname':model_hash,
                                               'args':feature_config})

    return {'child_jobs':jobids}
    
    
FEATURE_CACHE = {}

def get_from_cache(obj,cache):
    hash = hashlib.sha1(repr(obj)).hexdigest()
    if hash in cache:
        return cache[hash]
        
def put_in_cache(obj,value,cache):
    hash = hashlib.sha1(repr(obj)).hexdigest()
    cache[hash] = value

def compute_features(image_config, image_fs, model_config, model_fs,convolve_func):
    image_fh = image_fs.get_version(image_config['filename'])
    filter_fh = model_fs.get_version(model_config['filename'])
    return compute_features_core(image_fh,image_config,filter_fh,model_config,convolve_func)

def compute_features_core(image_fh,image_config,filter_fh,model_config,convolve_func):

    cached_val = get_from_cache((image_config,model_config),FEATURE_CACHE)
    if cached_val is not None:
        output = cached_val
    else:
        print('extracting', image_config, model_config)
     
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
        
        put_in_cache((image_config,model_config),pooled,FEATURE_CACHE)
        
        output = pooled
    
    return output
    
 

def evaluate_protocol(evaluate_config_path,feature_config_path,model_config_path,image_config_path,write=False):
    
    model_config_gen = get_config(model_config_path)
    image_config_gen = get_config(image_config_path)
    
    overall_config_gen = SON([('models',model_config_gen['models']),('images',image_config_gen['images'])])
    feature_hash = get_config_string(overall_config_gen)
    feature_certificate = '../.feature_certificates/' + feature_hash   
 
    evaluate_config = get_config(evaluate_config_path)
    task_config = evaluate_config.pop('train_test')
    
    D = []
    for task in task_config:
        overall_config_gen = SON([('models',model_config_gen),('image',image_config_gen),('task',task)])
        ext_hash = get_config_string(overall_config_gen)
        outfile = '../.performance_certificates/' + ext_hash
        op = ('svm_evaluation_' + ext_hash,evaluate,(outfile,feature_certificate,evaluate_config_path,task,ext_hash))
        D.append(op)

    if write:
        actualize(D)
    return D

STATS = ['test_accuracy','ap','auc','mean_ap','mean_auc','train_accuracy']   

@activate(lambda x : (x[1],x[2]),lambda x : x[0])
def evaluate(outfile,feature_certificate,cpath,task,ext_hash):

    conn = pm.Connection(document_class=bson.SON)
    db = conn[DB_NAME]
    
    perf_fs = gridfs.GridFS(db,'performance')
    perf_coll = db['performance.files']
    
    remove_existing(perf_coll,perf_fs,ext_hash)

    feature_certdict = cPickle.load(open(feature_certificate))
    feature_hash = feature_certdict['feature_hash']
    image_hash = feature_certdict['image_hash']
    model_hash = feature_certdict['model_hash']
    image_config_gen = feature_certdict['args']['images']
    model_col = db['models.files']
    feature_fs = gridfs.GridFS(db,'features')
    feature_col = db['features.files']
    
    stats = ['test_accuracy','ap','auc','mean_ap','mean_auc','train_accuracy']    
       
    if isinstance(task,list):
        task_list = task
    else:
        task_list = [task]
    
    model_configs = get_most_recent_files(model_col,{'__hash__':model_hash})
    
    for m in model_configs:
        print('Evaluating model',m) 
        for task in task_list:
            task['universe'] = task.get('universe',SON([]))
            task['universe']['model'] = m['config']['model']
            print('task', task)
            classifier_kwargs = task.get('classifier_kwargs',{})    
            split_results = []
            splits = generate_splits(task,feature_hash,'features') 
            for (ind,split) in enumerate(splits):
                print ('split', ind)
                train_data = split['train_data']
                test_data = split['test_data']
                
                train_filenames = [t['filename'] for t in train_data]
                test_filenames = [t['filename'] for t in test_data]
                assert set(train_filenames).intersection(test_filenames) == set([])
                
                print('train feature extraction ...')
                train_features = sp.row_stack([transform_average(load_features(f['filename'],feature_fs), task.get('transform_average'), m) for f in train_data])
                print('test feature extraction ...')
                test_features = sp.row_stack([transform_average(load_features(f['filename'],feature_fs), task.get('transform_average'), m) for f in test_data])
                train_labels = split['train_labels']
                test_labels = split['test_labels']
    
                print('classifier ...')
                res = svm.classify(train_features,train_labels,test_features,test_labels,classifier_kwargs)
                print('Split test accuracy', res['test_accuracy'])
                split_results.append(res)
        
            model_results = SON([])
            for stat in STATS:
                if stat in split_results[0] and split_results[0][stat] != None:
                    model_results[stat] = sp.array([split_result[stat] for split_result in split_results]).mean()           
    
            out_record = SON([('model',m['config']['model']),
                              ('model_hash',model_hash), 
                              ('model_filename',m['filename']), 
                              ('images',son_escape(image_config_gen)),
                              ('image_hash',image_hash),
                              ('task',son_escape(task)),
                         ])
                                             
            filename = get_filename(out_record)
            out_record['filename'] = filename
            out_record['config_path'] = cpath
            out_record['__hash__'] = ext_hash
            out_record.update(model_results)
            print('dump out ...')
            out_data = cPickle.dumps(SON([('split_results',split_results),('splits',splits)]))
            
            perf_fs.put(out_data,**out_record)

    createCertificateDict(outfile,{'feature_file':feature_certificate})
    
    
def load_features(filename,fs):
    cached_val = get_from_cache(filename,FEATURE_CACHE)
    if cached_val:
        return cached_val
    else:
        val = cPickle.loads(fs.get_version(filename).read())
        put_in_cache(filename,val,FEATURE_CACHE)
        return val
    

def extract_and_evaluate_protocol(evaluate_config_path,model_config_path,image_config_path,convolve_func_name='numpy', write=False):
    
    model_config_gen = get_config(model_config_path)
    model_hash = get_config_string(model_config_gen['models'])
    model_certificate = '../.model_certificates/' + model_hash
    
    image_config_gen = get_config(image_config_path)
    image_hash =  get_config_string(image_config_gen['images'])
    image_certificate = '../.image_certificates/' + image_hash

    evaluate_config = get_config(evaluate_config_path)
    task_config = evaluate_config.pop('train_test')

     
    D = [] 
    for task in task_config:
        overall_config_gen = SON([('models',model_config_gen),('images',image_config_gen),('task',task)])
        ext_hash = get_config_string(overall_config_gen)    
        
        parallel = task.pop('generate_parallel',False)
        if not parallel:
            performance_certificate = '../.performance_certificates/' + ext_hash
            op = ('evaluation_' + ext_hash,extract_and_evaluate,(performance_certificate,image_certificate,model_certificate,evaluate_config_path,convolve_func_name,task,ext_hash))
            D.append(op)
        else:
            performance_certificate = '../.performance_certificates/' + ext_hash
            batch_certificate = '../.batch_certificates/' + ext_hash
            op1 = ('launch_batches' + ext_hash,extract_and_evaluate_launch_batches,(batch_certificate,image_certificate,model_certificate,evaluate_config_path,convolve_func_name,task,ext_hash))
            D.append(op1)
            op2 = ('combine_splits' + ext_hash,extract_and_evaluate_combine_splits,(performance_certificate,batch_certificate,image_certificate,model_certificate,evaluate_config_path,ext_hash))
            D.append(op2)
             
    if write:
        actualize(D)
    return D
    
    
@activate(lambda x : (x[1],x[2],x[3]),lambda x : x[0])
def extract_and_evaluate(outfile,image_certificate_file,model_certificate_file,cpath,convolve_func_name,task,ext_hash):

    conn = pm.Connection(document_class=bson.SON)
    db = conn[DB_NAME]
    
    perf_fs = gridfs.GridFS(db,'performance')
    perf_coll = db['performance.files']
    
    remove_existing(perf_coll,perf_fs,ext_hash)

    print('using image certificate', image_certificate_file)
    print('using model certificate', model_certificate_file)   
    
    model_certdict = cPickle.load(open(model_certificate_file))
    model_hash = model_certdict['model_hash']
    model_coll = db['models.files']
    model_fs = gridfs.GridFS(db,'models')
    
    image_certdict = cPickle.load(open(image_certificate_file))
    image_hash = image_certdict['image_hash']
    image_config_gen = image_certdict['args']['images']
    image_fs = gridfs.GridFS(db,'images')
    
    stats = ['test_accuracy','ap','auc','mean_ap','mean_auc','train_accuracy']    
    
    model_configs = get_most_recent_files(model_coll,{'__hash__' : model_hash})

    if convolve_func_name == 'numpy':
        convolve_func = v1f.v1like_filter_numpy
    elif convolve_func_name == 'pyfft':
        convolve_func = v1f.v1like_filter_pyfft
    else:
        raise ValueError, 'convolve func name not recognized'    
    
    if convolve_func == v1f.v1like_filter_pyfft:
        v1_pyfft.setup_pyfft()
    
    if isinstance(task,list):
        task_list = task
    else:
        task_list = [task]
    
    for m in model_configs: 
        print('Evaluating model',m)
        for task in task_list:
            classifier_kwargs = task.get('classifier_kwargs',{})    
            print('task',task)
            split_results = []
            splits = generate_splits(task,image_hash,'images') 
            for (ind,split) in enumerate(splits):
                print ('split', ind)
                train_data = split['train_data']
                test_data = split['test_data']
                
                train_filenames = [t['filename'] for t in train_data]
                test_filenames = [t['filename'] for t in test_data]
                assert set(train_filenames).intersection(test_filenames) == set([])
                
                print('train feature extraction ...')
                train_features = sp.row_stack([transform_average(compute_features(im, image_fs, m, model_fs, convolve_func,) , task.get('transform_average'),m) for im in train_data])
                print('test feature extraction ...')
                test_features = sp.row_stack([transform_average(compute_features(im, image_fs, m, model_fs, convolve_func) , task.get('transform_average'),m) for im in test_data])
                train_labels = split['train_labels']
                test_labels = split['test_labels']
    
                print('classifier ...')
                res = svm.classify(train_features,train_labels,test_features,test_labels,classifier_kwargs)
                print('Split test accuracy', res['test_accuracy'])
                split_results.append(res)
        
            model_results = SON([])
            for stat in STATS:
                if stat in split_results[0] and split_results[0][stat] != None:
                    model_results[stat] = sp.array([split_result[stat] for split_result in split_results]).mean()           
    
            out_record = SON([('model',m['config']['model']),
                              ('model_hash',model_hash), 
                              ('model_filename',m['filename']), 
                              ('images',son_escape(image_config_gen)),
                              ('image_hash',image_hash),
                              ('task',son_escape(task)),
                         ])
 
            filename = get_filename(out_record)
            out_record['filename'] = filename
            out_record['config_path'] = cpath
            out_record.update(model_results)
            out_record['__hash__'] = ext_hash
            print('dump out ...')
            out_data = cPickle.dumps(SON([('split_results',split_results),('splits',splits)]))
            
            perf_fs.put(out_data,**out_record)
 
    if convolve_func == v1f.v1like_filter_pyfft:
        v1_pyfft.cleanup_pyfft() 
      
    createCertificateDict(outfile,{'image_file':image_certificate_file,'models_file':model_certificate_file})

        
def generate_splits(task_config,hash,colname):
    
    base_query = SON([('__hash__',hash)])
    ntrain = task_config['ntrain']
    ntest = task_config['ntest']
    ntrain_pos = task_config.get('ntrain_pos')
    ntest_pos = task_config.get('ntest_pos')
    N = task_config.get('N',10)
    query = task_config['query']  
    base_query.update(reach_in('config',task_config.get('universe',SON([]))))    
    cquery = reach_in('config',query)
    
    print('q',cquery)
    print('u',base_query)
 
    return traintest.generate_split2(DB_NAME,colname,cquery,N,ntrain,ntest,ntrain_pos=ntrain_pos,ntest_pos = ntest_pos,universe=base_query,use_negate = True)


 
import numpy as np
def transform_average(input,config,model_config):
    if config:
        averaged = []
        K = input.keys()
        K.sort()
        for cidx in K:
            averaged.append(average_transform(input[cidx],config,model_config))
        averaged = sp.concatenate(averaged)
        print(averaged)
        return averaged
    return input

def average_transform(input,config,M):
    if config['transform_name'] == 'translation':
        return input.sum(1).sum(0)
    elif config['transform_name'] == 'translation_and_orientation':
        model_config = M['config']['model'] 
        assert model_config.get('filter') and model_config['filter']['model_name'] == 'gridded_gabor'
        H = input.sum(1).sum(0) 
        norients = model_config['filter']['norients']
        phases = model_config['filter']['phases']
        nphases = len(phases)
        divfreqs = model_config['filter']['divfreqs']
        nfreqs = len(divfreqs)
        
        output = np.zeros((H.shape[0]/norients,)) 
        
        for freq_num in range(nfreqs):
            for phase_num in range(nphases):
                for orient_num in range(norients):
                    output[nphases*freq_num + phase_num] += H[norients*nphases*freq_num + nphases*orient_num + phase_num]
        
        return output
    elif config['transform_name'] == 'nothing':
        return input.ravel()
    elif config['transform_name'] == 'translation_and_fourier':
        return np.abs(np.fft.fft(input.sum(1).sum(0)))
    else:
        raise ValueError, 'Transform ' + str(config['transform_name']) + ' not recognized.'




from starflow.utils import ListUnion

def get_extraction_and_evaluation_batches(split_list):
    return ListUnion([[[[eval_id,m,task,ind,split]] for (ind,split) in enumerate(splits)] for (eval_id,m,task,splits) in split_list])


@activate(lambda x : (x[1],x[2],x[3]),lambda x : x[0])
def extract_and_evaluate_parallel_launch_batches(batch_certificate_file,image_certificate_file,model_certificate_file,cpath,convolve_func_name,task,ext_hash):

    conn = pm.Connection(document_class=bson.SON)
    db = conn[DB_NAME]
    
    batch_col = db['split_batches']
    batch_col.delete({'__hash__':ext_hash})
    
    split_col = db['performance_splits.files']
    split_fs = gridfs.GridFS(db,'performance_splits')
    remove_existing(split_col,split_fs,ext_hash)
    
    print('using image certificate', image_certificate_file)
    print('using model certificate', model_certificate_file)   
    
    model_certdict = cPickle.load(open(model_certificate_file))
    model_hash = image_certdict['model_hash']
    model_coll = db['models.files']
    model_fs = gridfs.GridFS(db,'models')
    
    image_certdict = cPickle.load(open(image_certificate_file))
    image_hash = image_certdict['image_hash']
    image_config_gen = image_certdict['args']['images']
    image_fs = gridfs.GridFS(db,'images')
    
    model_configs = get_most_recent_files(model_coll,{'__hash__' : model_hash})
    
    if isinstance(task,list):
        task_list = task
    else:
        task_list = [task]
    
    split_list = []
    for m in model_configs: 
        print('Evaluating model',m)
        for task in task_list:
            classifier_kwargs = task.get('classifier_kwargs',{})    
            task['universe'] = task.get('universe',SON([]))
            task['universe']['__hash__'] = image_hash

            splits = generate_splits_2(task,'images')
            eval_id = generate_splits_id()         
            split_list.append((eval_id,m,task,splits))
        
    batches = get_extraction_and_evaluation_batches(split_list)
    
    jobids = []
    
    for (id,batch) in enumerate(batches):
        batch_coll.insert({'__hash__':ext_hash,'batch_id':id,'batch':batch})
        if convolve_func_name == 'pyfft':
            devnum = (ind  %  numgpus) if (ngpus is not None) else None
        else:
            devnum = None


        jobid = qsub(extract_and_evaluate_parallel_run_batch,(ext_hash,id,convolve_func_name),queueName = 'feature_extraction')
        jobids.append(jobid)
    
    
    createCertificateDict(batch_certificate_file,{'image_file':image_certificate_file,'models_file':model_certificate_file})

    return {'child_jobs':jobids}
    
    
def extract_and_evaluate_parallel_run_batch(ext_hash,batch_id,convolve_func_name):

    conn = pm.Connection(document_class=bson.SON)
    db = conn[DB_NAME]
    
    batch_col = db['split_batches']
    batch_rec = batch_col.find_one({'__hash__':ext_hash,'batch_id':batch_id})
    batch = batch_rec['batch']
 
    model_fs = gridfs.GridFS(db,'models')
    image_fs = gridfs.GridFS(db,'images')
    
    split_fs = gridfs.GridFS(db,'performance_splits')
   
    if convolve_func_name == 'numpy':
        convolve_func = v1f.v1like_filter_numpy
    elif convolve_func_name == 'pyfft':
        convolve_func = v1f.v1like_filter_pyfft
    else:
        raise ValueError, 'convolve func name not recognized'

    device_id = get_device_id()
    if convolve_func == v1f.v1like_filter_pyfft:
        device_id = get_device_id()
        v1_pyfft.setup_pyfft(device_id = device_id) 
 
    for (eval_id,task,m,split_id, split) in batch:
            
        classifier_kwargs = task.get('classifier_kwargs',{}) 
        train_data = split['train_data']
        test_data = split['test_data']
        
        train_filenames = [t['filename'] for t in train_data]
        test_filenames = [t['filename'] for t in test_data]
        assert set(train_filenames).intersection(test_filenames) == set([])
        
        print('train feature extraction ...')
        train_features = sp.row_stack([transform_average(compute_features(im, image_fs, m, model_fs, convolve_func) , task.get('transform_average'),m) for im in train_data])
        print('test feature extraction ...')
        test_features = sp.row_stack([transform_average(compute_features(im, image_fs, m, model_fs, convolve_func) , task.get('transform_average'),m) for im in test_data])
        train_labels = split['train_labels']
        test_labels = split['test_labels']

        print('classifier ...')
        res = svm.classify(train_features,train_labels,test_features,test_labels,classifier_kwargs)
        print('Split test accuracy', res['test_accuracy'])
        
        rec = SON([('eval_id',eval_id), ('task',task), ('model',m), ('split_id', split_id)])
        
        results = SON([])
        for stat in STATS:
            if stat in res[0] and res[0][stat] != None:
                results[stat] = res[stat]
                
        filename = get_filename(rec)
        rec['results'] = results
        rec['filename'] = filename
        rec['__hash__'] = ext_hash
        
        out = cPickle.dumps({'results':res,'split':split})
        split_fs.put(out,**rec)
        
    
    if convolve_func == v1f.v1like_filter_pyfft:
        v1_pyfft.cleanup_pyfft() 
    
    
@activate(lambda x : (x[1],x[2],x[3],x[4]),lambda x : x[0])  
def extract_and_evaluate_parallel_combine_splits(outfile,batch_certificate_file,image_certificate_file,model_certificate_file,cpath,ext_hash):

    conn = pm.Connection(document_class=bson.SON)
    db = conn[DB_NAME]

    split_col = db['performance_splits.files']
    split_fs = gridfs.GridFS(db,'performance_splits')
    
    model_certdict = cPickle.load(open(model_certificate_file))
    model_hash = image_certdict['model_hash']
    model_coll = db['models.files']
    model_fs = gridfs.GridFS(db,'models')
    
    image_certdict = cPickle.load(open(image_certificate_file))
    image_hash = image_certdict['image_hash']
    image_config_gen = image_certdict['args']['images']
    image_fs = gridfs.GridFS(db,'images')
    
    perf_col = db['performance.files']
    perf_fs = gridfs.GridFS(db,'performance')

    splits = get_most_recent_files(split_col,{'__hash__':ext_hash})
    results = {}
    split_datas = []
    for split in splits:
        eval_id = split['eval_id']
        if not eval_id in results:
            results[eval_id] = {}
            results[eval_id]['task'] = split['task']
            results[eval_id]['model'] = split['model']
            
        if 'results' in results[eval_id]:
            results[eval_id]['results'].append(split['results'])
        else:
            results[eval_id]['results'] = split['results']
        
        if not eval_id in split_datas:
            split_data[eval_id] = {}
        split_data = cPickle.loads(split_fs.get_version(split['filename']).read())
        split_id = split['split_id']
        split_datas[eval_id]['results'][split_id] = split_data['results']
        split_datas[eval_id]['splits'][split_id] = split_data['split']
        
    
    for eval_id in results:
        R = results[eval_id].pop('results')
        for k in R[0]:
            results[eval_id][k] = np.array([r[k] for r in R]).mean()
        
        out_record = SON([('model',results[eval_id]['model']['config']['model']),
                          ('model_hash',model_hash), 
                          ('model_filename',results[eval_id]['model']['filename']), 
                          ('images',son_escape(image_config_gen)),
                          ('image_hash',image_hash),
                          ('task',son_escape(task)),
                     ])

        filename = get_filename(out_record)
        out_record['filename'] = filename
        out_record['config_path'] = cpath
        out_record.update(results[eval_id])
        out_record['__hash__'] = ext_hash
        print('dump out ...')
        out_data = cPickle.dumps(SON([('split_results',split_datas['eval_id']['results']),('splits',split_datas['eval_id']['splits'])]))
        
        perf_fs.put(out_data,**out_record)
        
        
    createCertificateDict(outfile,{'image_file':image_certificate_file,'models_file':model_certificate_file})
