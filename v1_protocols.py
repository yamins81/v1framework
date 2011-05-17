import sys
import cPickle
import hashlib
import os
import random
import multiprocessing
import functools 

import scipy as sp
import numpy as np
import pymongo as pm
import gridfs
from bson import SON
import bson
import zmq

from starflow.protocols import protocolize, actualize
from starflow.utils import activate
from starflow.sge_utils import wait_and_get_statuses

import v1like_extract as v1e
import v1like_funcs as v1f
import traintest
import svm
import rendering
import filter_generation
import starflow.de as de
from starflow.utils import ListUnion

from dbutils import get_config_string, get_filename, reach_in, createCertificateDict, son_escape, get_most_recent_files
from v1like_extract import get_config
from sge_utils import qsub

from pythor_networking import NETWORK_CACHE_PORT


DB_NAME = 'v1-test'

try:
    import pycuda.driver as cuda
except:
    GPU_SUPPORT = False
else:
    import v1_pyfft
    GPU_SUPPORT = True

def remove_existing(coll,fs, hash):
    existing = coll.find({'__hash__':hash})
    for e in existing:
        fs.delete(e['_id'])


def image_protocol(config_path,write = False,parallel=False):

    config = get_config(config_path)

    image_hash = get_config_string(config['images'])
    image_certificate = '../.image_certificates/' + image_hash

    if  not parallel:
        D = [('generate_images',generate_images,(image_certificate,image_hash,config))]
    else:
        D = [('generate_images',generate_images_parallel,(image_certificate,image_hash,config))]
    
    if write:
        actualize(D)
    return D,image_hash


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
        x['image']['generator'] = config_gen['images']['generator'] 
        jobid = qsub(generate_and_insert_single_image,(x,im_hash),opstring='-pe orte 2 -l qname=rendering.q -o /home/render -e /home/render')  
        jobids.append(jobid)
        
    createCertificateDict(outfile,{'image_hash':im_hash,'args':config_gen})

    return {'child_jobs':jobids}
    
   
def model_config_generator(config): 
    models = config['models']
    return [SON([('model',m)]) for m in models]   
    
def model_protocol(config_path,write = False,parallel=False):

    config = get_config(config_path)

    model_hash = get_config_string(config['models'])
    model_certificate = '../.model_certificates/' + model_hash

    D = [('generate_models',generate_models,(model_certificate,model_hash,config))]
    
    if write:
        actualize(D)
    return D,model_hash


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


def extract_features_protocol(image_config_path,
                              model_config_path,
                              convolve_func_name = 'numpy',
                              write = False,
                              parallel=False,
                              batch_size=1000):

    model_config_gen = get_config(model_config_path)
    model_hash = get_config_string(model_config_gen['models'])
    model_certificate = '../.model_certificates/' + model_hash
    
    image_config_gen = get_config(image_config_path)
    image_hash = get_config_string(image_config_gen['images'])
    image_certificate = '../.image_certificates/' + image_hash
    
    overall_config_gen = SON([('models',model_config_gen['models']),('images',image_config_gen['images'])])
    feature_hash = get_config_string(overall_config_gen)
    feature_certificate = '../.feature_certificates/' + feature_hash
       
    if not parallel:
        D = [('extract_features',
              extract_features,(feature_certificate,
                                image_certificate,
                                model_certificate,
                                overall_config_gen,
                                feature_hash,
                                convolve_func_name))]
    else:
        D = [('extract_features',
              extract_features_parallel,(feature_certificate,
                                         image_certificate,
                                         model_certificate,
                                         overall_config_gen,
                                         feature_hash,
                                         convolve_func_name,
                                         batch_size))]
    
    if write:
        actualize(D)
    return D, feature_hash


@activate(lambda x : (x[1],x[2]), lambda x : x[0])    
def extract_features(feature_certificate,
                     image_certificate,
                     model_certificate,
                     feature_config,
                     feature_hash,
                     convolve_func_name):

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

    extract_features_core(image_certificate,
                          model_certificate,
                          feature_hash,
                          image_hash,
                          model_hash,
                          convolve_func_name)
     
    createCertificateDict(feature_certificate,{'feature_hash':feature_hash,
                                               'image_hash':image_hash,
                                               'model_hash':model_hash,
                                               'args':feature_config,})

@activate(lambda x : (x[1],x[2]), lambda x : x[0])    
def extract_features_parallel(feature_certificate,
                              image_certificate,
                              model_certificate,
                              feature_config,
                              feature_hash,
                              convolve_func_name,
                              batch_size):
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
    
    limits = get_feature_batches(image_hash,model_hash,db['images.files'],db['models.files'],batch_size = batch_size)
    
    jobids = []

    if convolve_func_name == 'numpy':
        queueName = 'extraction_cpu.q'
    elif convolve_func_name == 'pyfft':
        queueName = 'extraction_gpu.q'

    opstring = '-l qname=' + queueName + ' -o /home/render -e /home/render'
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
                                              'm_limit':m_to-m_from
                                              }],
                                            opstring=opstring)
        jobids.append(jobid)

    createCertificateDict(feature_certificate,{'feature_hash':feature_hash,
                                               'image_hash':image_hash,
                                               'model_hash':model_hash,
                                               'args':feature_config})

    return {'child_jobs':jobids}


def get_num_gpus():
    p = multiprocessing.Pool(1)
    r = p.apply(get_num_gpus_core,())
    return r

def get_num_gpus_core():
    cuda.init()
    num = 0
    while True:
        try:
            cuda.Device(num)
        except:
            break
        else:
            num +=1
    return num


def extract_features_core(image_certificate,
                          model_certificate,
                          feature_hash,
                          image_hash,
                          model_hash,
                          convolve_func_name,
                          im_query=None,
                          m_query=None,im_skip=0,im_limit=0,m_skip=0,m_limit=0):

    if convolve_func_name == 'numpy':
        num_batches = multiprocessing.cpu_count()
        if num_batches > 1:
            pool = multiprocessing.Pool()
    elif convolve_func_name == 'pyfft':
        num_batches = get_num_gpus()
        if num_batches > 1:
            pool = multiprocessing.Pool(processes = num_batches)
        else:
            pool = None
    else:
        raise ValueError, 'convolve func name not recognized'

    if num_batches == 1:
        extract_features_inner_core(image_certificate,
                          model_certificate,
                          feature_hash,
                          image_hash,
                          model_hash,
                          convolve_func_name,
                          0,
                          im_query,
                          m_query,im_skip,im_limit,m_skip,m_limit)
    else:
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

        batches = get_feature_batches(image_hash,model_hash,image_col,model_col,im_skip=im_skip,
                                      im_limit = im_limit, m_skip = m_skip, 
                                      m_limit = m_limit, num_batches=num_batches)

        print('batches',batches)
        res = []
        for (batch_num,(s0,l0,s1,l1)) in enumerate(batches):
            args = (image_certificate,
                          model_certificate,
                          feature_hash,
                          image_hash,
                          model_hash,
                          convolve_func_name,
                          batch_num,
                          im_query,m_query,s0,l0,s1,l1)
            res.append(pool.apply_async(extract_features_inner_core,args))

        finished = [r.get() for r in res]

def extract_features_inner_core(image_certificate, model_certificate, feature_hash, image_hash,
     model_hash, convolve_func_name,device_id, im_query, m_query, im_skip,
     im_limit, m_skip, m_limit):

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
                       
    if convolve_func_name == 'pyfft':
        context = v1_pyfft.setup_pyfft(device_id)
        context.push()
        convolve_func = functools.partial(v1f.v1like_filter_pyfft,device_id=device_id)
    else:
        convolve_func = v1f.v1like_filter_numpy

    L1 = get_most_recent_files(image_col,im_query,skip=im_skip,limit=im_limit)
    L2 = get_most_recent_files(model_col,m_query,skip=m_skip,limit=m_limit)
        
    for image_config in L1:
        for model_config in L2: 
            features = compute_features(image_config['filename'], image_fs, model_config, model_fs,convolve_func)
            features_string = cPickle.dumps(features)
            y = SON([('config',SON([('model',model_config['config']['model']),('image',image_config['config']['image'])]))])
            filename = get_filename(y['config'])
            y['filename'] = filename
            y['__hash__'] = feature_hash
            feature_fs.put(features_string,**y)
            
            
    if convolve_func_name == 'pyfft':
        context.pop()
            
        
def get_feature_batches(im_hash,m_hash,im_col,m_col,im_skip=0,im_limit = 0, m_skip = 0, m_limit = 0, batch_size=None,num_batches=None):
    im_count = im_col.find({'__hash__':im_hash}).skip(im_skip).limit(im_limit).count(True)
    m_count = m_col.find({'__hash__':m_hash}).skip(m_skip).limit(m_limit).count(True)

    print im_count, num_batches, im_skip,im_limit,m_skip,m_limit

    if batch_size:
        im_batches = [(batch_size*i,min(batch_size*(i+1),im_count)) for i in range(int(math.ceil(float(im_count)/batch_size)))]
    else:
        batch_size = int(math.ceil(float(im_count)/num_batches))
        im_batches = [(batch_size*i,min(batch_size*(i+1),im_count)) for i in range(num_batches)]
    
    m_batches = [(j,j+1) for j in range(m_count)]
    
    return [(imb[0]+im_skip,imb[1]+im_skip,mb[0]+m_skip,mb[1]+m_skip) for imb in im_batches for mb in m_batches]
    

def compute_features(image_filename, image_fs, model_config, model_fs,convolve_func):
    image_fh = image_fs.get_version(image_filename)
    filter_fh = model_fs.get_version(model_config['filename'])
    print('extracting', image_filename, model_config)
    return compute_features_core(image_fh,filter_fh,model_config,convolve_func)

        
def compute_features_core(image_fh,filter_fh,model_config,convolve_func):


 
    m_config = model_config['config']['model']
    conv_mode = m_config['conv_mode']
    
    #preprocessing
    array = v1e.image2array(m_config ,image_fh)
  
    preprocessed,orig_imga = v1e.preprocess(array,m_config)
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
    
    return pooled
    
    
def evaluate_protocol(evaluate_config_path,model_config_path,image_config_path,write=False):
    
    model_config_gen = get_config(model_config_path)
    image_config_gen = get_config(image_config_path)
    
    overall_config_gen = SON([('models',model_config_gen['models']),
                              ('images',image_config_gen['images'])])
    feature_hash = get_config_string(overall_config_gen)
    feature_certificate = '../.feature_certificates/' + feature_hash   
 
    evaluate_config = get_config(evaluate_config_path)
    task_config = evaluate_config.pop('train_test')
    
    D = []
    ext_hashes = []
    for task in task_config:
        overall_config_gen = SON([('models',model_config_gen),
                                  ('image',image_config_gen),('task',task)])
        ext_hash = get_config_string(overall_config_gen)
        outfile = '../.performance_certificates/' + ext_hash
        op = ('svm_evaluation_' + ext_hash,evaluate,(outfile,feature_certificate,
                                            evaluate_config_path,task,ext_hash))
        D.append(op)
        ext_hashes.append(ext_hash)
    if write:
        actualize(D)
    return D,ext_hashes

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
                train_features = sp.row_stack([load_features(f['filename'],feature_fs,m,task) for f in train_data])
                print('test feature extraction ...')
                test_features = sp.row_stack([load_features(f['filename'],feature_fs,m,task) for f in test_data])
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


FEATURE_CACHE = {}

def get_from_cache(obj,cache):
    hash = hashlib.sha1(repr(obj)).hexdigest()
    if hash in cache:
        return cache[hash]
        
def put_in_cache(obj,value,cache):
    hash = hashlib.sha1(repr(obj)).hexdigest()
    cache[hash] = value
    
    
def load_features(filename,fs,m,task):
    cached_val = get_from_cache((filename,task.get('transform_average')),FEATURE_CACHE)
    if cached_val is not None:
        return cached_val
    else:
        val = transform_average(cPickle.loads(fs.get_version(filename).read()),task.get('transform_average'),m)
        put_in_cache((filename,task.get('transform_average')),val,FEATURE_CACHE)
        return val
    

def get_features(im,im_fs,m,m_fs,convolve_func,task,network_cache):

    if network_cache:
        obj =(im,m,task.get('transform_average')) 
        hash = hashlib.sha1(repr(obj)).hexdigest()
        network_cache.send_pyobj({'get':hash})
        val = network_cache.recv_pyobj()
        if val is not None:
            output = val
        else:
            output = transform_average(compute_features(im, im_fs, m, m_fs, convolve_func) , task.get('transform_average'),m)
            network_cache.send_pyobj({'put':(hash,output)})
            network_cache.recv_pyobj()
    else:
        output = transform_average(compute_features(im, im_fs, m, m_fs, convolve_func) , task.get('transform_average'),m)
    return output
    

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

   
def extract_and_evaluate_inner_core(images,m,convolve_func_name,device_id,task,cache_port):

    if cache_port:
        ctx = zmq.Context()
        sock = ctx.socket(zmq.REQ)
        sock.connect('tcp://127.0.0.1:' + str(cache_port))  
    else:
        sock = None

    if convolve_func_name == 'pyfft':
        context = v1_pyfft.setup_pyfft(device_id)
        context.push()
        convolve_func = functools.partial(v1f.v1like_filter_pyfft,device_id=device_id)
    else:
        convolve_func = v1f.v1like_filter_numpy

    conn = pm.Connection(document_class=bson.SON)
    db = conn[DB_NAME]

    perf_coll = db['performance.files']

    model_fs = gridfs.GridFS(db,'models')
    image_fs = gridfs.GridFS(db,'images')

    L = [get_features(im, image_fs, m, model_fs, convolve_func,task,sock) for im in images]
    
    if convolve_func_name == 'pyfft':
        context.pop()
        
    return L

import math

def get_data_batches(data,num_batches):

    bs = int(math.ceil(float(len(data)) / num_batches))
    
    return [data[bs*i:bs*(i+1)] for i in range(num_batches)]
    
def extract_and_evaluate_core(split,m,convolve_func_name,task,cache_port):
    classifier_kwargs = task.get('classifier_kwargs',{})  
    train_data = split['train_data']
    test_data = split['test_data']
    train_labels = split['train_labels']
    test_labels = split['test_labels']                
    train_filenames = [t['filename'] for t in train_data]
    test_filenames = [t['filename'] for t in test_data]
    assert set(train_filenames).intersection(test_filenames) == set([])

    existing_train_features = [get_from_cache((tf,m,task.get('transform_average')),FEATURE_CACHE) for tf in train_filenames]
    existing_train_labels = [train_labels[i] for (i,x) in enumerate(existing_train_features) if x is not None]
    new_train_filenames = [train_filenames[i] for (i,x) in enumerate(existing_train_features) if x is None]
    new_train_labels = [train_labels[i] for (i,x) in enumerate(existing_train_features) if x is None]


    existing_test_features = [get_from_cache((tf,m,task.get('transform_average')),FEATURE_CACHE) for tf in test_filenames]
    existing_test_labels = [test_labels[i] for (i,x) in enumerate(existing_test_features) if x is not None]
    new_test_filenames =[test_filenames[i] for (i,x) in enumerate(existing_test_features) if x is None]
    new_test_labels = [test_labels[i] for (i,x) in enumerate(existing_test_features) if x is None]

    if convolve_func_name == 'numpy':
        num_batches = multiprocessing.cpu_count()
        if num_batches > 1:
            pool = multiprocessing.Pool()
    elif convolve_func_name == 'pyfft':

        num_batches = get_num_gpus()
        if num_batches > 1:
            pool = multiprocessing.Pool(processes = num_batches)
        else:
            pool = None
    else:
        raise ValueError, 'convolve func name not recognized'

    if num_batches > 1:
        batches = get_data_batches(new_train_filenames,num_batches)
        results = []
        for (bn,b) in enumerate(batches):
            results.append(pool.apply_async(extract_and_evaluate_inner_core,(b,m.to_dict(),convolve_func_name,bn,task.to_dict(),cache_port)))
        results = [r.get() for r in results]
        new_train_features = ListUnion(results)
        batches = get_data_batches(new_test_filenames,num_batches)
        results = []
        for (bn,b) in enumerate(batches):
            results.append(pool.apply_async(extract_and_evaluate_inner_core,(b,m.to_dict(),convolve_func_name,bn,task.to_dict(),cache_port)))
        results = [r.get() for r in results]
        new_test_features = ListUnion(results)
    else:
        print('train feature extraction ...')
        new_train_features = extract_and_evaluate_inner_core(new_train_filenames,m,convolve_func_name,0,task,cache_port)
        print('test feature extraction ...')
        new_test_features = extract_and_evaluate_inner_core(new_test_filenames,m,convolve_func_name,0,task,cache_port)

    #TODO get the order consistent with original ordering
    train_features = sp.row_stack(filter(lambda x : x is not None,existing_train_features) + new_train_features)
    test_features = sp.row_stack(filter(lambda x : x is not None, existing_test_features) + new_test_features)
    train_labels = existing_train_labels + new_train_labels
    test_labels = existing_test_labels + new_test_labels
    
    for (im,f) in zip(new_train_filenames,new_train_features):
        put_in_cache((im,m,task.get('transform_average')),f,FEATURE_CACHE)
    for(im,f) in zip(new_test_filenames,new_test_features):
        put_in_cache((im,m,task.get('transform_average')),f,FEATURE_CACHE)
                           
    print('classifier ...')
    res = svm.classify(train_features,train_labels,test_features,test_labels,classifier_kwargs)
    print('Split test accuracy', res['test_accuracy'])
    return res

     
def prepare_extract_and_evaluate(ext_hash,image_certificate_file,model_certificate_file,task):

    conn = pm.Connection(document_class=bson.SON)
    db = conn[DB_NAME]
    

    perf_coll = db['performance']
    perf_coll.remove({'__hash__':ext_hash})
    split_coll = db['splits.files']
    split_fs = gridfs.GridFS(db,'splits')
    remove_existing(split_coll,split_fs,ext_hash)
    splitperf_coll = db['split_performance.files']
    splitperf_fs = gridfs.GridFS(db,'split_performance')
    remove_existing(splitperf_coll,splitperf_fs,ext_hash)

    model_certdict = cPickle.load(open(model_certificate_file))
    model_hash = model_certdict['model_hash']
    model_coll = db['models.files']
    
    image_certdict = cPickle.load(open(image_certificate_file))
    image_hash = image_certdict['image_hash']
    image_config_gen = image_certdict['args']
    model_configs = get_most_recent_files(model_coll,{'__hash__' : model_hash})
    
    if isinstance(task,list):
        task_list = task
    else:
        task_list = [task]
    
    return model_configs,image_config_gen,model_hash,image_hash, task_list, perf_coll, split_coll, split_fs, splitperf_coll, splitperf_fs
    

def put_in_performance(split_results,image_config_gen,m,model_hash,image_hash,perf_coll,task,ext_hash):
    
    model_results = SON([])
    for stat in STATS:
        if stat in split_results[0] and split_results[0][stat] != None:
            model_results[stat] = sp.array([split_result[stat] for split_result in split_results]).mean()           

    out_record = SON([('model',m['config']['model']),
                      ('model_hash',model_hash), 
                      ('model_filename',m['filename']), 
                      ('images',son_escape(image_config_gen['images'])),
                      ('image_hash',image_hash),
                      ('task',son_escape(task)),
                      ('__hash__',ext_hash)
                 ])
                 
    out_record.update(model_results)

    perf_coll.insert(out_record)


def put_in_split(split,image_config_gen,m,task,ext_hash,split_id,split_fs):
    out_record = SON([('model',m['config']['model']),
                      ('images',son_escape(image_config_gen['images'])),
                      ('task',son_escape(task)),
                      ('split_id',split_id),
                 ])   

    filename = get_filename(out_record)
    out_record['filename'] = filename
    out_record['__hash__'] = ext_hash
    print('dump out ...')
    out_data = cPickle.dumps(SON([('split',split)]))
    
    split_fs.put(out_data,**out_record)
            
def put_in_split_result(res,image_config_gen,m,task,ext_hash,split_id,splitres_fs):
    out_record = SON([('model',m['config']['model']),
                      ('images',son_escape(image_config_gen['images'])),
                      ('task',son_escape(task)),
                      ('split_id',split_id),
                 ])   
                 
    split_result = SON([])
    for stat in STATS:
        if stat in res and res[stat] != None:
            split_result[stat] = res[stat] 

    filename = get_filename(out_record)
    out_record['filename'] = filename
    out_record['__hash__'] = ext_hash
    out_record.update(split_result)
    print('dump out split result...')
    out_data = cPickle.dumps(SON([('split_result',res)]))
    splitres_fs.put(out_data,**out_record)          
 

@activate(lambda x : (x[1],x[2],x[3]),lambda x : x[0])
def extract_and_evaluate(outfile,image_certificate_file,model_certificate_file,cpath,convolve_func_name,task,ext_hash):

    (model_configs, image_config_gen, model_hash, image_hash, task_list, 
    perf_col, split_coll, split_fs, splitperf_coll, splitperf_fs) = prepare_extract_and_evaluate(ext_hash,
                                                image_certificate_file,
                                                model_certificate_file,
                                                task)
    for m in model_configs: 
        print('Evaluating model',m)
        for task in task_list:  
            print('task',task)
            split_results = []
            splits = generate_splits(task,image_hash,'images') 
            for (ind,split) in enumerate(splits):
                put_in_split(split,image_config_gen,m,task,ext_hash,ind,split_fs)
                res = extract_and_evaluate_core(split,m,convolve_func_name,task,None)    
                put_in_split_result(res,image_config_gen,m,task,ext_hash,ind,splitperf_fs)
                split_results.append(res)
            put_in_performance(split_results,image_config_gen,m,model_hash,image_hash,perf_col,task,ext_hash)

        
    createCertificateDict(outfile,{'image_file':image_certificate_file,'models_file':model_certificate_file})


def extract_and_evaluate_parallel_core(image_config_gen,m,task,ext_hash,split_id,convolve_func_name,cache_port=None):

    if cache_port is None:
        cache_port = NETWORK_CACHE_PORT

               
    conn = pm.Connection(document_class=bson.SON)
    db = conn[DB_NAME]
    split_col = db['splits.files']
    split_fs = gridfs.GridFS(db,'splits')

    splitconf = get_most_recent_files(split_col,{'__hash__':ext_hash,'split_id':split_id,'model':m['config']['model'],'images':son_escape(image_config_gen['images'])})[0]
    split = cPickle.loads(split_fs.get_version(splitconf['filename']).read())['split']
    res = extract_and_evaluate_core(split,m,convolve_func_name,task,cache_port)
    splitperf_fs = gridfs.GridFS(db,'split_performance')
    put_in_split_result(res,image_config_gen,m,task,ext_hash,split_id,splitperf_fs)


@activate(lambda x : (x[1],x[2],x[3]),lambda x : x[0])
def extract_and_evaluate_parallel(outfile,image_certificate_file,model_certificate_file,cpath,convolve_func_name,task,ext_hash):
        
    (model_configs, image_config_gen, model_hash, image_hash, task_list,
     perf_col, split_coll, split_fs, splitperf_coll, splitperf_fs) = prepare_extract_and_evaluate(ext_hash,
                                                                                                  image_certificate_file,
                                                                                                  model_certificate_file,
                                                                                                  task)

    
    jobids = []
    if convolve_func_name == 'numpy':
        opstring = '-l qname=extraction_cpu.q'
    elif convolve_func_name == 'pyfft':
        opstring = '-l qname=extraction_gpu.q -o /home/render -e /home/render'
        
    for m in model_configs: 
        print('Evaluating model',m)
        for task in task_list:
            classifier_kwargs = task.get('classifier_kwargs',{})    
            print('task',task)
            splits = generate_splits(task,image_hash,'images') 
            for (ind,split) in enumerate(splits):
                put_in_split(split,image_config_gen,m,task,ext_hash,ind,split_fs)  
                jobid = qsub(extract_and_evaluate_parallel_core,(image_config_gen,m,task,ext_hash,ind,convolve_func_name),opstring=opstring)
                jobids.append(jobid)

    print(jobids)
    statuses = wait_and_get_statuses(jobids)
    
    for m in model_configs: 
        print('Evaluating model',m)
        for task in task_list:
            split_results = get_most_recent_files(splitperf_coll,{'__hash__':ext_hash,'task':son_escape(task),'model':m['config']['model'],'images':son_escape(image_config_gen['images'])})
            put_in_performance(split_results,image_config_gen,m,model_hash,image_hash,perf_col,task,ext_hash)

    createCertificateDict(outfile,{'image_file':image_certificate_file,'models_file':model_certificate_file})


def extract_and_evaluate_protocol(evaluate_config_path,model_config_path,image_config_path,convolve_func_name='numpy', write=False,parallel=False):
    
    model_config_gen = get_config(model_config_path)
    model_hash = get_config_string(model_config_gen['models'])
    model_certificate = '../.model_certificates/' + model_hash
    
    image_config_gen = get_config(image_config_path)
    image_hash =  get_config_string(image_config_gen['images'])
    image_certificate = '../.image_certificates/' + image_hash

    evaluate_config = get_config(evaluate_config_path)
    task_config = evaluate_config.pop('train_test')

    D = []
    DH = {}
    for task in task_config:
        overall_config_gen = SON([('models',model_config_gen),('images',image_config_gen),('task',task)])
        ext_hash = get_config_string(overall_config_gen)    
        
        if not parallel:
            performance_certificate = '../.performance_certificates/' + ext_hash
            op = ('evaluation_' + ext_hash,extract_and_evaluate,(performance_certificate,image_certificate,model_certificate,evaluate_config_path,convolve_func_name,task,ext_hash))
            D.append(op)
            DH[ext_hash] = [op]
        else:
            performance_certificate = '../.performance_certificates/' + ext_hash
            batch_certificate = '../.batch_certificates/' + ext_hash
            op = ('evaluation' + ext_hash,extract_and_evaluate_parallel,(batch_certificate,image_certificate,model_certificate,evaluate_config_path,convolve_func_name,task,ext_hash))
            D.append(op)
            DH[ext_hash] = [op]
             
    if write:
        actualize(D)
    return DH


        
