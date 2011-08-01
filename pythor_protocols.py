import sys
import cPickle
import hashlib
import os
import random
import multiprocessing
import functools 
import copy
import math

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

import v1like_funcs as v1f
import traintest
import svm
import rendering
import filter_generation
import starflow.de as de
from starflow.utils import ListUnion, uniqify
from processing import image2array, preprocess, postprocess

from dbutils import get_config_string, get_filename, reach_in, createCertificateDict, son_escape, get_most_recent_files
from sge_utils import qsub
from pythor_networking import NETWORK_CACHE_PORT, NETWORK_CACHE_TIMEOUT

DB_NAME = 'thor'

import pythor3.operation.lnorm_.plugins.cthor
import pythor3.operation.lpool_.plugins.cthor
import pythor3.operation.fbcorr_.plugins.cthor
import pythor3.operation.fbcorr_.plugins.numpyFFT


from pythor3.operation.fbcorr_ import (
    DEFAULT_STRIDE,
    DEFAULT_MIN_OUT,
    DEFAULT_MAX_OUT,
    DEFAULT_MODE)

try:
    import pycuda.driver as cuda
except:
    GPU_SUPPORT = False
else:
    import pythor3.operation.fbcorr_.plugins.cuFFT as cuFFT
    GPU_SUPPORT = True


FEATURE_CACHE = {}



#################IMAGES#############
#################IMAGES#############
#################IMAGES#############
#################IMAGES#############
#################IMAGES#############

def image_protocol_hash(config_path):
    config = get_config(config_path)
    image_hash = get_config_string(config['images'])
    return image_hash

def image_protocol(config_path,write = False,parallel=False):
    config = get_config(config_path) 
    image_hash = image_protocol_hash(config_path)
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
        jobid = qsub(generate_and_insert_single_image,(x,im_hash),opstring='-pe orte 2 -l qname=rendering.q -o /home/render/image_jobs -e /home/render/image_jobs')  
        jobids.append(jobid)
        
    createCertificateDict(outfile,{'image_hash':im_hash,'args':config_gen})

    return {'child_jobs':jobids}
    


#################MODELS#############
#################MODELS#############
#################MODELS#############
#################MODELS#############
#################MODELS#############

def model_protocol_hash(config_path):
    config = get_config(config_path)
    model_hash = get_config_string(config['models'])
    return model_hash
 
   
def model_config_generator(config): 
    models = config['models']
    return [SON([('model',m)]) for m in models]   
    
def model_protocol(config_path,write = False,parallel=False):

    config = get_config(config_path)
    model_hash = model_protocol_hash(config_path)
    model_certificate = '../.model_certificates/' + model_hash

    D = [('generate_models',generate_models,(model_certificate,model_hash,config))]
    
    if write:
        actualize(D)
    return D,model_hash

def get_model(m):
    filterbanks = filter_generation.get_hierarchical_filterbanks(m['layers']) 
    for (layer,filterbank) in zip(m['layers'],filterbanks):
        if layer.get('activ'):
            if layer['activ'].get('min_out_gen') == 'random':
                minmax = layer['activ']['min_out_max']
                minmin = layer['activ']['min_out_min']
                layer['activ']['min_out'] = list((minmax-minmin)*np.random.random(size=filterbank.shape[0]) + minmin)
            if layer['activ'].get('max_out_gen') == 'random':
                maxmax = layer['activ']['max_out_max']
                maxmin = layer['activ']['max_out_min']
                layer['activ']['max_out'] = list((maxmax-maxmin)*np.random.random(size=filterbank.shape[0]) + maxmin)
            if hasattr(layer['activ'].get('min_out'),'__iter__') and not hasattr(layer['activ'].get('max_out'),'__iter__'):
                layer['activ']['max_out'] = [layer['activ'].get('max_out')]*len(layer['activ']['min_out'])
            if hasattr(layer['activ'].get('max_out'),'__iter__') and not hasattr(layer['activ'].get('min_out'),'__iter__'):
                layer['activ']['min_out'] = [layer['activ'].get('min_out')]*len(layer['activ']['max_out'])
                        
    return filterbanks

@activate(lambda x : (), lambda x : x[0])    
def generate_models(outfile,m_hash,config_gen):

    conn = pm.Connection(document_class = SON)
    db = conn[DB_NAME]
    m_coll = db['models.files']
    m_fs = gridfs.GridFS(db,'models')
    
    remove_existing(m_coll,m_fs,m_hash)
    
    M = model_config_generator(config_gen)       
    
    for (i,m) in enumerate(M):
        if isinstance(m['model'],list):
            filterbanks = [get_model(model) for model in m['model']]
        else:
            filterbanks = get_model(m['model'])
            
        filterbank_string = cPickle.dumps(filterbanks)
        if (i/5)*5 == i:
            print(i,m) 
        
        y = SON([('config',m)])
        filename = get_filename(m)
        y['filename'] = filename
        y['__hash__'] = m_hash
        m_fs.put(filterbank_string,**y)
        
    createCertificateDict(outfile,{'model_hash':m_hash,'args':config_gen})
    
    

#################EXTRACTION#############
#################EXTRACTION#############
#################EXTRACTION#############
#################EXTRACTION#############

def extraction_protocol(extraction_config_path,model_config_path,image_config_path,
                        convolve_func_name='numpy', write=False,parallel=False,save_to_db=False,
                        batch_size=None):
                        
    model_config_gen = get_config(model_config_path)
    model_hash = get_config_string(model_config_gen['models'])
    model_certificate = '../.model_certificates/' + model_hash
    
    image_config_gen = get_config(image_config_path)
    image_hash =  get_config_string(image_config_gen['images'])
    image_certificate = '../.image_certificates/' + image_hash

    extraction_config = get_config(extraction_config_path)
    extraction_config = extraction_config.pop('extractions')

    D = []
    DH = {}
    for task in extraction_config:
        overall_config_gen = SON([('models',model_config_gen['models']),('images',image_config_gen['images']),('extraction',task)])
        ext_hash = get_config_string(overall_config_gen)    
        
        extraction_certificate = '../.extraction_certificates/' + ext_hash
        if parallel:
            func = extract_parallel
        else:
            func = extract
                                                
        op = ('evaluation_' + ext_hash,func, (extraction_certificate,
                                              image_certificate,
                                              model_certificate,
                                              extraction_config_path,
                                              convolve_func_name,
                                              task,
                                              ext_hash,
                                              save_to_db,batch_size))                                                
        D.append(op)
        DH[ext_hash] = [op]
             
    if write:
        actualize(D)
    return DH



@activate(lambda x : (x[1],x[2],x[3]),lambda x : x[0])
def extract(outfile,image_certificate_file,model_certificate_file,cpath,convolve_func_name,
            task, ext_hash, save_to_db,batch_size):

    (model_configs, model_hash, image_hash, task_list) = prepare_extract(ext_hash,
                                                                         image_certificate_file, 
                                                                         model_certificate_file, 
                                                                         task)
                    
    for m in model_configs: 
        print('Evaluating model',m)
        for task in task_list:  
            print('task',task)
            extract_core(image_hash,m,model_hash,task,ext_hash,convolve_func_name,save_to_db,None,None)    
        
    createCertificateDict(outfile,{'image_file':image_certificate_file,'models_file':model_certificate_file})
    

        
    
@activate(lambda x : (x[1],x[2],x[3]),lambda x : x[0])
def extract_parallel(outfile,
                     image_certificate_file,
                     model_certificate_file,
                     cpath,
                     convolve_func_name,
                     task,
                     ext_hash,
                     save_to_db,
                     batch_size):
        
    (model_configs, model_hash, image_hash, task_list) = prepare_extract(ext_hash,
                                                                         image_certificate_file, 
                                                                         model_certificate_file, 
                                                                         task)
    
#    if cache_port is None:
#        cache_port = NETWORK_CACHE_PORT
    cache_port = None    
    
    jobids = []
    if convolve_func_name == 'numpy':
        opstring = '-l qname=extraction_cpu.q -o /home/render -e /home/render'
    elif convolve_func_name == 'cufft':
        opstring = '-l qname=extraction_gpu.q -o /home/render -e /home/render'
        
    for m in model_configs: 
        print('Evaluating model',m)
        for task in task_list:
            print('task',task)
            batches = get_extraction_batches(image_hash,task,batch_size)
            for batch in batches:
                jobid = qsub(extract_core,(image_hash,m,model_hash,task,ext_hash,convolve_func_name,save_to_db,batch,cache_port),opstring=opstring)
                print('Submitted job', jobid)
                jobids.append(jobid)
                
    print('Waiting for jobs', jobids) 
    statuses = wait_and_get_statuses(jobids)
    
    if not all([status == 0 for status in statuses]):
        bad_jobs = [jobid for (jobid,status) in zip(jobids,statuses) if not status == 0]
        raise ValueError, 'There was a error in job(s): ' + repr(bad_jobs)

    createCertificateDict(outfile,{'image_file':image_certificate_file,'models_file':model_certificate_file})
    
 
def extract_core(image_hash,m,model_hash,task,ext_hash,convolve_func_name,save_to_db,batch,cache_port):

    conn = pm.Connection(document_class=bson.SON)
    db = conn[DB_NAME]
    
    image_configs = list(get_extraction_configs(image_hash,task,batch))
    filenames = map(lambda x : x['filename'],image_configs)
    
    existing_features = [get_from_cache((tf,m,task.get('transform_average')),FEATURE_CACHE) for tf in filenames]
    existing_filenames = [filenames[i] for (i,x) in enumerate(existing_features) if x is not None]
    new_filenames = [filenames[i] for (i,x) in enumerate(existing_features) if x is None]
    existing_configs = [image_configs[i] for (i,x) in enumerate(existing_features) if x is not None]
    new_configs = [image_configs[i] for (i,x) in enumerate(existing_features) if x is None]
    
    filenames_reordered = existing_filenames + new_filenames
    configs_reordered = existing_configs + new_configs

    if convolve_func_name == 'numpy':
        num_inner_batches = multiprocessing.cpu_count()
        if num_inner_batches > 1:
            print('found %d processors, using that many processes' % num_inner_batches)
            pool = multiprocessing.Pool(num_inner_batches)
            print('allocated pool')
        else:
            pool = multiprocessing.Pool(1)
    elif convolve_func_name == 'cufft':
        num_inner_batches = get_num_gpus()
        if num_inner_batches > 1:
            print('found %d gpus, using that many processes' % num_inner_batches)
            pool = multiprocessing.Pool(processes = num_inner_batches)
        else:
            pool = multiprocessing.Pool(1)
    else:
        raise ValueError, 'convolve func name not recognized'

    print('num_batches',num_inner_batches)
    if num_inner_batches > 0:
        inner_batches = get_data_batches(new_filenames,num_inner_batches)
        results = []
        for (bn,b) in enumerate(inner_batches):
            results.append(pool.apply_async(extract_inner_core,(b,m.to_dict(),convolve_func_name,bn,task.to_dict(),cache_port)))
        results = [r.get() for r in results]
        new_features = ListUnion(results)
    else:
        new_features = extract_inner_core(new_filenames,m,convolve_func_name,0,task,cache_port,save_to_db)

    features = filter(lambda x : x is not None,existing_features) + new_features
    
    for (im,f) in zip(new_filenames,new_features):
        put_in_cache((im,m,task.get('transform_average')),f,FEATURE_CACHE)
       
    put_in_extraction(features,configs_reordered,m,model_hash,image_hash,task,ext_hash,save_to_db)


def extract_inner_core(images,m,convolve_func_name,device_id,task,cache_port):

    if cache_port:
        ctx = zmq.Context()
        sock = ctx.socket(zmq.REQ)
        sock.connect('tcp://127.0.0.1:' + str(cache_port))  
        sock.send_pyobj({'alive':True})
        poller = zmq.Poller()
        poller.register(sock)
        poll = poller.poll(timeout=NETWORK_CACHE_TIMEOUT)
        if poll != []:
            sock.recv_pyobj()
        else:
            poller = None
    else:
        poller = None

    if convolve_func_name == 'cufft':
        convolve_func = cuFFT.LFBCorrCuFFT(device_id=device_id, use_cache=True)
        context = convolve_func.context
    else:
        convolve_func = c_numpy_mixed
        

    conn = pm.Connection(document_class=bson.SON)
    db = conn[DB_NAME]

    perf_coll = db['performance.files']

    model_fs = gridfs.GridFS(db,'models')
    image_fs = gridfs.GridFS(db,'images')

    filter_fh = model_fs.get_version(m['filename'])
    filter = cPickle.loads(filter_fh.read())
    
    L = [get_features(im, image_fs, filter, m, convolve_func,task,poller) for im in images]
    
    if convolve_func_name == 'cufft':
        context.pop()
        
    return L

def get_features(im,im_fs,filter,m,convolve_func,task,network_cache):

    if network_cache and network_cache.sockets:
        sock = network_cache.sockets.keys()[0]
        obj = (im,m,task.get('transform_average')) 
        hash = hashlib.sha1(repr(obj)).hexdigest()
        sock.send_pyobj({'get':hash})
        poll = network_cache.poll(timeout=NETWORK_CACHE_TIMEOUT)
        if poll != []:
            val = sock.recv_pyobj()
        else:
            val = None
            network_cache.unregister(sock)
        if val is not None:
            output = val
        else:
            output = transform_average(compute_features(im, im_fs, filter, m, convolve_func) , task.get('transform_average'),m)
            sock.send_pyobj({'put':(hash,output)})
            sock.recv_pyobj()
    else:
        output = transform_average(compute_features(im, im_fs, filter, m, convolve_func) , task.get('transform_average'),m)
    return output
    


def put_in_extraction(features,image_configs,m,model_hash,image_hash,task,ext_hash,save_to_db):
    out_record = SON([('model',m['config']['model']),
                      ('model_hash',model_hash), 
                      ('model_filename',m['filename']), 
                      ('image_hash',image_hash),
                      ('extraction',son_escape(task)),
                 ])   

    conn = pm.Connection(document_class=bson.SON)
    db = conn[DB_NAME]
    feature_fs = gridfs.GridFS(db,'features')
    
    for (feat,config) in zip(features,image_configs):
        out_record = copy.deepcopy(out_record)
        out_record['image'] = config['config']['image']
        out_record['image_filename'] = config['filename']
        
        out_record['filename'] = get_filename(out_record)
        out_record['__hash__'] = ext_hash
        if save_to_db:
            out_record['feature'] = feat.tolist()
            out_record['feature_length'] = len(feat)
        print('pickling split result...')
        out_data = cPickle.dumps(feat)
        print('dumping out split result ...')
        feature_fs.put(out_data,**out_record)          
                      

def prepare_extract(ext_hash,image_certificate_file,model_certificate_file,task):

    conn = pm.Connection(document_class=bson.SON)
    db = conn[DB_NAME]
    

    feature_coll = db['features.files']
    feature_fs = gridfs.GridFS(db,'features')
    remove_existing(feature_coll,feature_fs,ext_hash) 

    model_certdict = cPickle.load(open(model_certificate_file))
    model_hash = model_certdict['model_hash']
    model_coll = db['models.files']
    
    image_certdict = cPickle.load(open(image_certificate_file))
    image_hash = image_certdict['image_hash']
    model_configs = get_most_recent_files(model_coll,{'__hash__' : model_hash})
    
    if isinstance(task,list):
        task_list = task
    else:
        task_list = [task]
    
    return model_configs,model_hash,image_hash, task_list
    




#################EVALUATION#############
#################EVALUATION#############
#################EVALUATION#############
#################EVALUATION#############
#################EVALUATION#############    
    
STATS = ['test_accuracy','ap','auc','mean_ap','mean_auc','train_accuracy']  

def evaluate_protocol(evaluation_config_path,extraction_config_path,model_config_path,image_config_path,
                      convolve_func_name='numpy', write=False,parallel=False):
                        
    model_config_gen = get_config(model_config_path)
    model_hash = get_config_string(model_config_gen['models'])
    model_certificate = '../.model_certificates/' + model_hash
    
    image_config_gen = get_config(image_config_path)
    image_hash =  get_config_string(image_config_gen['images'])
    image_certificate = '../.image_certificates/' + image_hash

    extraction_config = get_config(extraction_config_path)
    extraction_config = extraction_config.pop('extractions')

    evaluation_config = get_config(evaluation_config_path)
    evaluation_config = evaluation_config.pop('train_test')

    D = []
    DH = {}
    for extraction in extraction_config:
        extraction_config_gen = SON([('models',model_config_gen['models']),('images',image_config_gen['images']),('extraction',extraction)])
        extraction_hash = get_config_string(extraction_config_gen)
        extraction_certificate = '../.extraction_certificates/' + extraction_hash
        
        for evaluation in evaluation_config:
            overall_config_gen = SON([('models',model_config_gen['models']),('images',image_config_gen['images']),('extraction',extraction),('train_test',evaluation)])
            ext_hash = get_config_string(overall_config_gen)    
            
            performance_certificate = '../.performance_certificates/' + ext_hash
            if not parallel:
                func = evaluate
            else:
                func = evaluate_parallel
                                                    
            op = ('evaluation_' + ext_hash,func, (performance_certificate,
                                                  extraction_certificate, 
                                                  image_certificate,
                                                  model_certificate,
                                                  convolve_func_name,
                                                  evaluation,
                                                  ext_hash))                                                
            D.append(op)
            DH[ext_hash] = [op]
             
    if write:
        actualize(D)
    return DH


@activate(lambda x : (x[1],x[2],x[3]),lambda x : x[0])
def evaluate(outfile,extraction_certificate_file,image_certificate_file,model_certificate_file,convolve_func_name,task,ext_hash):

    (model_configs, image_config_gen, model_hash, image_hash, task_list, 
    perf_col, split_coll, split_fs, splitperf_coll, splitperf_fs) = prepare_evaluate(ext_hash,
                                                image_certificate_file,
                                                model_certificate_file,
                                                task)
    for m in model_configs: 
        print('Evaluating model',m)
        for task in task_list:  
            print('task',task)
            split_results = []
            splits = generate_splits(task,image_hash,'images',overlap=task.get('overlap')) 
            for (ind,split) in enumerate(splits):
                put_in_split(split,image_config_gen,m,task,ext_hash,ind,split_fs)
                print('evaluating split %d' % ind)
                res = evaluate_core(split,m,convolve_func_name,task)    
                put_in_split_result(res,image_config_gen,m,task,ext_hash,ind,splitperf_fs)
                split_results.append(res)
            put_in_performance(split_results,image_config_gen,m,model_hash,image_hash,perf_col,task,ext_hash)

        
    createCertificateDict(outfile,{'image_file':image_certificate_file,'models_file':model_certificate_file,'extraction_file':extraction_certificate_file})


@activate(lambda x : (x[1],x[2],x[3]),lambda x : x[0])
def evaluate_parallel(outfile,extraction_certificate,image_certificate_file,model_certificate_file,convolve_func_name,task,ext_hash):
        
    (model_configs, image_config_gen, model_hash, image_hash, task_list,
     perf_col, split_coll, split_fs, splitperf_coll, splitperf_fs) = prepare_evaluate(ext_hash,
                                                                                                  image_certificate_file,
                                                                                                  model_certificate_file,
                                                                                                  task)

    
    jobids = []
    if convolve_func_name == 'numpy':
        opstring = '-l qname=extraction_cpu.q -o /home/render -e /home/render'
    elif convolve_func_name == 'cufft':
        opstring = '-l qname=extraction_gpu.q -o /home/render -e /home/render'
    
    for task in task_list:
        splits = generate_splits(task,image_hash,'images',overlap=task.get('overlap')) 
        for m in model_configs: 
            print('Evaluating model',m)
            print('On task',task)              
            for (ind,split) in enumerate(splits):
                put_in_split(split,image_config_gen,m,task,ext_hash,ind,split_fs)  
            jobid = qsub(evaluate_parallel_core,
                         (image_config_gen,m,task,ext_hash,convolve_func_name),
                         opstring=opstring)
            print('Submitted job', jobid)
            jobids.append(jobid)
                
    print('Waiting for jobs', jobids) 
    statuses = wait_and_get_statuses(jobids)
    
    if not all([status == 0 for status in statuses]):
        bad_jobs = [jobid for (jobid,status) in zip(jobids,statuses) if not status == 0]
        raise ValueError, 'There was a error in job(s): ' + repr(bad_jobs)    
    
    for m in model_configs: 
        print('Evaluating model',m)
        for task in task_list:
            split_results = get_most_recent_files(splitperf_coll,{'__hash__':ext_hash,'task':son_escape(task),'model':m['config']['model'],'images':son_escape(image_config_gen['images'])})
            put_in_performance(split_results,image_config_gen,m,model_hash,image_hash,perf_col,task,ext_hash)

    createCertificateDict(outfile,{'image_file':image_certificate_file,'models_file':model_certificate_file,'extraction_file':extraction_certificate_file})

def evaluate_parallel_core(image_config_gen,m,task,ext_hash,split_id,convolve_func_name):

    conn = pm.Connection(document_class=bson.SON)
    db = conn[DB_NAME]
    split_col = db['splits.files']
    split_fs = gridfs.GridFS(db,'splits')

    splitconf = get_most_recent_files(split_col,{'__hash__':ext_hash,
                                                 'split_id':split_id,
                                                 'model':m['config']['model'],
                                                 'task':son_escape(task),
                                                 'images':son_escape(image_config_gen['images'])})[0]
    split = cPickle.loads(split_fs.get_version(splitconf['filename']).read())['split']
    res = evaluate_core(split,m,convolve_func_name,task)
    splitperf_fs = gridfs.GridFS(db,'split_performance')
    put_in_split_result(res,image_config_gen,m,task,ext_hash,split_id,splitperf_fs)


def evaluate_core(split,m,convolve_func_name,task):
    classifier_kwargs = task.get('classifier_kwargs',{})  
    train_data = split['train_data']
    test_data = split['test_data']
    train_labels = split['train_labels']
    test_labels = split['test_labels']                
    train_filenames = [t['filename'] for t in train_data]
    test_filenames = [t['filename'] for t in test_data]
    assert set(train_filenames).intersection(test_filenames) == set([])

    conn = pm.Connection(document_class=bson.SON)
    db = conn[DB_NAME]
    feature_coll = db['features.files']
    feature_fs = gridfs.GridFS(db,'features')
    
    print('train feature extraction ...')
    train_features = sp.row_stack([load_features(f,feature_coll,feature_fs,m,task) for f in train_filenames])
    print('test feature extraction ...')
    test_features = sp.row_stack([load_features(f,feature_coll,feature_fs,m,task) for f in test_filenames])
    train_labels = split['train_labels']
    test_labels = split['test_labels']          
    
    print('classifier ...')
    if len(uniqify(train_labels + test_labels)) > 2:
        #res = svm.ova_classify(train_features,train_labels,test_features,test_labels,classifier_kwargs)
        res = svm.multi_classify(train_features,train_labels,test_features,test_labels,**classifier_kwargs)
    else:
        res = svm.classify(train_features,train_labels,test_features,test_labels,classifier_kwargs)
    print('Split test accuracy', res['test_accuracy'])
    return res
    
    
def load_features(image_filename,coll,fs,m,task):
    filename = coll.find_one({'model':m['config']['model'],'image_filename':image_filename},fields=["filename"])["filename"]
    return cPickle.loads(fs.get_version(filename).read())

def prepare_evaluate(ext_hash,image_certificate_file,model_certificate_file,task):
    return prepare_extract_and_evaluate(ext_hash,image_certificate_file,model_certificate_file,task)


#########EXTRACT AND EVALUATE############# 
#########EXTRACT AND EVALUATE############# 
#########EXTRACT AND EVALUATE############# 
#########EXTRACT AND EVALUATE############# 
#########EXTRACT AND EVALUATE############# 

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
            print('found %d processors, using that many processes' % num_batches)
            pool = multiprocessing.Pool(num_batches)
            print('allocated pool')
        else:
            pool = multiprocessing.Pool(1)
    elif convolve_func_name == 'cufft':
        num_batches = get_num_gpus()
        #num_batches = 1
        if num_batches > 1:
            print('found %d gpus, using that many processes' % num_batches)
            pool = multiprocessing.Pool(processes = num_batches)
        else:
            pool = multiprocessing.Pool(1)
    else:
        raise ValueError, 'convolve func name not recognized'

    print('num_batches',num_batches)
    if num_batches > 0:
        batches = get_data_batches(new_train_filenames,num_batches)
        results = []
        for (bn,b) in enumerate(batches):
            results.append(pool.apply_async(extract_inner_core,(b,m.to_dict(),convolve_func_name,bn,task.to_dict(),cache_port)))
        results = [r.get() for r in results]
        new_train_features = ListUnion(results)
        batches = get_data_batches(new_test_filenames,num_batches)
        results = []
        for (bn,b) in enumerate(batches):
            results.append(pool.apply_async(extract_inner_core,(b,m.to_dict(),convolve_func_name,bn,task.to_dict(),cache_port)))
        results = [r.get() for r in results]
        new_test_features = ListUnion(results)
    else:
        print('train feature extraction ...')
        new_train_features = extract_inner_core(new_train_filenames,m,convolve_func_name,0,task,cache_port)
        print('test feature extraction ...')
        new_test_features = extract_inner_core(new_test_filenames,m,convolve_func_name,0,task,cache_port)

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
    if len(uniqify(train_labels + test_labels)) > 2:
        #res = svm.ova_classify(train_features,train_labels,test_features,test_labels,classifier_kwargs)
        res = svm.multi_classify(train_features,train_labels,test_features,test_labels,**classifier_kwargs)
    else:
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

    print('inserting result ...')
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
    print('pickling split ...')
    out_data = cPickle.dumps(SON([('split',split)]))
    print('dump out split ...')
    split_fs.put(out_data,**out_record)

import bson           
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
    print('pickling split result...')
    out_data = cPickle.dumps(SON([('split_result',res)]))
    print('dumping out split result ...')
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
                print('evaluating split %d' % ind)
                res = extract_and_evaluate_core(split,m,convolve_func_name,task,None)    
                put_in_split_result(res,image_config_gen,m,task,ext_hash,ind,splitperf_fs)
                split_results.append(res)
            put_in_performance(split_results,image_config_gen,m,model_hash,image_hash,perf_col,task,ext_hash)

        
    createCertificateDict(outfile,{'image_file':image_certificate_file,'models_file':model_certificate_file})


def extract_and_evaluate_parallel_core(image_config_gen,m,task,ext_hash,split_id,convolve_func_name,cache_port=None):

    if cache_port is None:
        cache_port = NETWORK_CACHE_PORT
    cache_port = None
                       
    conn = pm.Connection(document_class=bson.SON)
    db = conn[DB_NAME]
    split_col = db['splits.files']
    split_fs = gridfs.GridFS(db,'splits')

    splitconf = get_most_recent_files(split_col,{'__hash__':ext_hash,
                                                 'split_id':split_id,
                                                 'model':m['config']['model'],
                                                 'task':son_escape(task),
                                                 'images':son_escape(image_config_gen['images'])})[0]
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
        opstring = '-l qname=extraction_cpu.q -o /home/render -e /home/render'
    elif convolve_func_name == 'cufft':
        opstring = '-l qname=extraction_gpu.q -o /home/render -e /home/render'
        
    for m in model_configs: 
        print('Evaluating model',m)
        for task in task_list:
            print('task',task)
            splits = generate_splits(task,image_hash,'images') 
            for (ind,split) in enumerate(splits):
                put_in_split(split,image_config_gen,m,task,ext_hash,ind,split_fs)  
                jobid = qsub(extract_and_evaluate_parallel_core,(image_config_gen,m,task,ext_hash,ind,convolve_func_name),opstring=opstring)
                print('Submitted job', jobid)
                jobids.append(jobid)
                
    print('Waiting for jobs', jobids) 
    statuses = wait_and_get_statuses(jobids)
    
    if not all([status == 0 for status in statuses]):
        bad_jobs = [jobid for (jobid,status) in zip(jobids,statuses) if not status == 0]
        raise ValueError, 'There was a error in job(s): ' + repr(bad_jobs)
    
    
    for m in model_configs: 
        print('Evaluating model',m)
        for task in task_list:
            split_results = get_most_recent_files(splitperf_coll,{'__hash__':ext_hash,'task':son_escape(task),'model':m['config']['model'],'images':son_escape(image_config_gen['images'])})
            put_in_performance(split_results,image_config_gen,m,model_hash,image_hash,perf_col,task,ext_hash)

    createCertificateDict(outfile,{'image_file':image_certificate_file,'models_file':model_certificate_file})

def extract_and_evaluate_semi_parallel_core(image_config_gen,m,task,ext_hash,convolve_func_name,cache_port=None):

    if cache_port is None:
        cache_port = NETWORK_CACHE_PORT
    cache_port = None
        
               
    conn = pm.Connection(document_class=bson.SON)
    db = conn[DB_NAME]
    split_col = db['splits.files']
    split_fs = gridfs.GridFS(db,'splits')

    splitconfs = get_most_recent_files(split_col,{'__hash__':ext_hash,
                                                  'model':m['config']['model'],
                                                  'task':son_escape(task),
                                                  'images':son_escape(image_config_gen['images'])})
    for splitconf in splitconfs:
        split = cPickle.loads(split_fs.get_version(splitconf['filename']).read())['split']
        split_id = splitconf['split_id']
        res = extract_and_evaluate_core(split,m,convolve_func_name,task,cache_port)
        splitperf_fs = gridfs.GridFS(db,'split_performance')
        put_in_split_result(res,image_config_gen,m,task,ext_hash,split_id,splitperf_fs)


@activate(lambda x : (x[1],x[2],x[3]),lambda x : x[0])
def extract_and_evaluate_semi_parallel(outfile,image_certificate_file,model_certificate_file,cpath,convolve_func_name,task,ext_hash):
        
    (model_configs, image_config_gen, model_hash, image_hash, task_list,
     perf_col, split_coll, split_fs, splitperf_coll, splitperf_fs) = prepare_extract_and_evaluate(ext_hash,
                                                                                                  image_certificate_file,
                                                                                                  model_certificate_file,
                                                                                                  task)

    
    jobids = []
    if convolve_func_name == 'numpy':
        opstring = '-l qname=extraction_cpu.q -o /home/render -e /home/render'
    elif convolve_func_name == 'cufft':
        opstring = '-l qname=extraction_gpu.q -o /home/render -e /home/render'
    
    for task in task_list:
        splits = generate_splits(task,image_hash,'images',overlap=task.get('overlap')) 
        for m in model_configs: 
            print('Evaluating model',m)
            print('On task',task)              
            for (ind,split) in enumerate(splits):
                put_in_split(split,image_config_gen,m,task,ext_hash,ind,split_fs)  
            jobid = qsub(extract_and_evaluate_semi_parallel_core,
                         (image_config_gen,m,task,ext_hash,convolve_func_name),
                         opstring=opstring)
            print('Submitted job', jobid)
            jobids.append(jobid)
                
    print('Waiting for jobs', jobids) 
    statuses = wait_and_get_statuses(jobids)
    
    if not all([status == 0 for status in statuses]):
        bad_jobs = [jobid for (jobid,status) in zip(jobids,statuses) if not status == 0]
        raise ValueError, 'There was a error in job(s): ' + repr(bad_jobs)    
    
    for m in model_configs: 
        print('Evaluating model',m)
        for task in task_list:
            split_results = get_most_recent_files(splitperf_coll,{'__hash__':ext_hash,'task':son_escape(task),'model':m['config']['model'],'images':son_escape(image_config_gen['images'])})
            put_in_performance(split_results,image_config_gen,m,model_hash,image_hash,perf_col,task,ext_hash)

    createCertificateDict(outfile,{'image_file':image_certificate_file,'models_file':model_certificate_file})


def get_extract_and_evaluate_hashes(evaluate_config_path,model_config_path,image_config_path):
    model_config_gen = get_config(model_config_path)
    model_hash = get_config_string(model_config_gen['models'])
    model_certificate = '../.model_certificates/' + model_hash
    
    image_config_gen = get_config(image_config_path)
    image_hash =  get_config_string(image_config_gen['images'])
    image_certificate = '../.image_certificates/' + image_hash

    evaluate_config = get_config(evaluate_config_path)
    task_config = evaluate_config.pop('train_test')

    DH = []
    for task in task_config:
        overall_config_gen = SON([('models',model_config_gen),('images',image_config_gen),('task',task)])
        ext_hash = get_config_string(overall_config_gen)    
        DH.append(ext_hash)
    
    return DH
             
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
        
        performance_certificate = '../.performance_certificates/' + ext_hash
        if not parallel:
            func = extract_and_evaluate
        elif parallel == 'semi':
            func = extract_and_evaluate_semi_parallel
        else:
            func = extract_and_evaluate_parallel
                                                
        op = ('evaluation_' + ext_hash,func, (performance_certificate,
                                              image_certificate,
                                              model_certificate,
                                              evaluate_config_path,
                                              convolve_func_name,
                                              task,
                                              ext_hash))                                                
        D.append(op)
        DH[ext_hash] = [op]
             
    if write:
        actualize(D)
    return DH
    


#################NATURAL STATISTICS###########
#################NATURAL STATISTICS###########
#################NATURAL STATISTICS###########
#################NATURAL STATISTICS###########
#################NATURAL STATISTICS###########

def get_corr_protocol(corr_config_path,model_config_path,image_config_path,convolve_func_name='numpy', write=False,parallel=False):
    model_config_gen = get_config(model_config_path)
    model_hash = get_config_string(model_config_gen['models'])
    model_certificate = '../.model_certificates/' + model_hash
    
    image_config_gen = get_config(image_config_path)
    image_hash =  get_config_string(image_config_gen['images'])
    image_certificate = '../.image_certificates/' + image_hash

    corr_config = get_config(corr_config_path)
    task_config = corr_config.pop('extractions')

    D = []
    DH = {}
    for task in task_config:
        overall_config_gen = SON([('models',model_config_gen),('images',image_config_gen),('task',task)])
        ext_hash = get_config_string(overall_config_gen)    
        
        performance_certificate = '../.corr_extraction_certificates/' + ext_hash
        if not parallel:
            op = ('extraction_' + ext_hash,get_corr,(performance_certificate,image_certificate,model_certificate,corr_config_path,convolve_func_name,task,ext_hash))
        else:
            op = ('extraction_' + ext_hash,get_corr_parallel,(performance_certificate,image_certificate,model_certificate,corr_config_path,convolve_func_name,task,ext_hash))
        D.append(op)
        DH[ext_hash] = [op]
             
    if write:
        actualize(D)
    return DH

    
    
@activate(lambda x : (x[1],x[2],x[3]),lambda x : x[0])
def get_corr_parallel(outfile,image_certificate_file,model_certificate_file,cpath,convolve_func_name,task,ext_hash):
        
    (model_configs, image_config_gen, model_hash, image_hash, task_list,
     sample_coll, sample_fs, extraction_coll, extraction_fs) = prepare_corr(ext_hash,
                                                              image_certificate_file,
                                                              model_certificate_file,
                                                              task)
    
    jobids = []
    if convolve_func_name == 'numpy':
        opstring = '-l qname=extraction_cpu.q -o /home/render -e /home/render'
    elif convolve_func_name == 'cufft':
        opstring = '-l qname=extraction_gpu.q -o /home/render -e /home/render'
        
    for m in model_configs: 
        print('Evaluating model',m)
        for task in task_list:
            classifier_kwargs = task.get('classifier_kwargs',{})    
            print('task',task)
            sample = generate_random_sample(task,image_hash,'images') 
            put_in_sample(sample,image_config_gen,m,task,ext_hash,sample_fs)  
            jobid = qsub(get_corr_parallel_core,(image_config_gen,m,task,ext_hash,model_hash,image_hash,convolve_func_name),opstring=opstring)
            print('Submitted job', jobid)
            jobids.append(jobid)
                
    print('Waiting for jobs', jobids) 
    statuses = wait_and_get_statuses(jobids)
    
    if not all([status == 0 for status in statuses]):
        bad_jobs = [jobid for (jobid,status) in zip(jobids,statuses) if not status == 0]
        raise ValueError, 'There was a error in job(s): ' + repr(bad_jobs)
    
    createCertificateDict(outfile,{'image_file':image_certificate_file,'models_file':model_certificate_file})
    
@activate(lambda x : (x[1],x[2],x[3]),lambda x : x[0])
def get_corr(outfile,image_certificate_file,model_certificate_file,cpath,convolve_func_name,task,ext_hash):

    (model_configs, image_config_gen, model_hash, image_hash, task_list, 
     sample_coll, sample_fs,extraction_coll,extraction_fs) = prepare_corr(ext_hash,
                                                image_certificate_file,
                                                model_certificate_file,
                                                task)
    for m in model_configs: 
        print('Evaluating model',m)
        for task in task_list:  
            print('task',task)
            sample = generate_random_sample(task,image_hash,'images') 
            put_in_sample(sample,image_config_gen,m,task,ext_hash,ind,sample_fs)  
            res = get_corr_core(sample,m,convolve_func_name,task,None)    
            put_in_sample_result(res,image_config_gen,m,task,ext_hash,model_hash,image_hash,extraction_fs)
    
    createCertificateDict(outfile,{'image_file':image_certificate_file,'models_file':model_certificate_file})
    
    
def get_corr_parallel_core(image_config_gen,m,task,ext_hash,model_hash,image_hash,convolve_func_name,cache_port=None):

    if cache_port is None:
        cache_port = NETWORK_CACHE_PORT
    cache_port = None
        

               
    conn = pm.Connection(document_class=bson.SON)
    db = conn[DB_NAME]
    sample_col = db['samples.files']
    sample_fs = gridfs.GridFS(db,'samples')

    sampleconf = get_most_recent_files(sample_col,{'__hash__':ext_hash,'model':m['config']['model'],'images':son_escape(image_config_gen['images'])})[0]
    sample = cPickle.loads(sample_fs.get_version(sampleconf['filename']).read())['sample']
    res = get_corr_core(sample,m,convolve_func_name,task,cache_port)
    extraction_fs = gridfs.GridFS(db,'correlation_extraction')
    put_in_sample_result(res,image_config_gen,m,task,ext_hash,model_hash,image_hash,extraction_fs)


def get_corr_core(sample,m,convolve_func_name,task,cache_port):

    sample_filenames = map(str,sample)

    if convolve_func_name == 'numpy':
        num_batches = multiprocessing.cpu_count()
        if num_batches > 1:
            print('found %d processors, using that many processes' % num_batches)
            pool = multiprocessing.Pool(num_batches)
            print('allocated pool')
        else:
            pool = multiprocessing.Pool(1)
    elif convolve_func_name == 'cufft':
        num_batches = get_num_gpus()
        #num_batches = 1
        if num_batches > 1:
            print('found %d gpus, using that many processes' % num_batches)
            pool = multiprocessing.Pool(processes = num_batches)
        else:
            pool = multiprocessing.Pool(1)
    else:
        raise ValueError, 'convolve func name not recognized'

    print('num_batches',num_batches)
    if num_batches > 0:
        batches = get_data_batches(sample_filenames,num_batches)
        results = []
        weights = []
        for (bn,b) in enumerate(batches):
            results.append(pool.apply_async(get_corr_inner_core,(b,m.to_dict(),convolve_func_name,bn,task.to_dict(),cache_port)))
            weights.append(len(b))
        weights = np.array(weights) / float(sum(weights))
        batch_extractions = [r.get() for r in results]
        extractions = combine_corr(batch_extractions,weights = weights)
    else:
        extractions = get_corr_inner_core(sample_filenames,m,convolve_func_name,0,task,cache_port)

    
    return extractions


def get_corr_inner_core(images,m,convolve_func_name,device_id,task,cache_port):

    if cache_port:
        ctx = zmq.Context()
        sock = ctx.socket(zmq.REQ)
        sock.connect('tcp://127.0.0.1:' + str(cache_port))  
        sock.send_pyobj({'alive':True})
        poller = zmq.Poller()
        poller.register(sock)
        poll = poller.poll(timeout=NETWORK_CACHE_TIMEOUT)
        if poll != []:
            sock.recv_pyobj()
        else:
            poller = None
    else:
        poller = None

    if convolve_func_name == 'cufft':
        convolve_func = cuFFT.LFBCorrCuFFT(device_id=device_id, use_cache=True)
        context = convolve_func.context
    else:
        convolve_func = c_numpy_mixed
        

    conn = pm.Connection(document_class=bson.SON)
    db = conn[DB_NAME]

    perf_coll = db['performance.files']

    model_fs = gridfs.GridFS(db,'models')
    image_fs = gridfs.GridFS(db,'images')

    filter_fh = model_fs.get_version(m['filename'])
    filters = cPickle.loads(filter_fh.read())
    cshape = filters[-1].shape[0]
    
    s = task['ker_shape']; size = s[0]*s[1]*cshape
    V = np.zeros((size,size))
    M = np.zeros((size,))
    task = copy.deepcopy(task)
    task['transform_average'] = {'transform_name':'central_slice','ker_shape':task['ker_shape']}
    for (n,im) in enumerate(images):
        f = get_features(im, image_fs, filters, m, convolve_func,task,poller)
        V,M = combine_corr([(V,M),(0,f)],np.array([1 - (1/(n+1)),1/(n+1)]))
    
    if convolve_func_name == 'cufft':
        context.pop()
        
    return (V,M)

def stack_channels(input):
    X = np.empty(input[0].shape + (len(input),),dtype=input[0].dtype)
    K = input.keys()
    K.sort()
    for (ind,k) in enumerate(K):
        X[ind] = input[k]
    return X


def get_central_slice(f,s):
    fshape = f.shape[:2]
    d0 = (fshape[0] - s[0])/2
    d1 = (fshape[1] - s[1])/2
    return f[d0:d0+s[0],d1:d1+s[1]].ravel()
    
outer = np.outer
def combine_corr(batches,weights=None):
    if weights is None:
        weights = np.ones(len(batches)) / len(batches)
    

    if len(batches) > 2:
        subweights = weights[:-1]
        subweights = subweights/sum(subweights)
        res1 = combine_corr(batches[:-1],weights=subweights) 
        res2 = batches[-1]
        w1 = sum(weights[:-1])
        w2 = weights[-1]
    else:
        res1 = batches[0]
        res2 = batches[-1]
        w1 = weights[0]
        w2 = weights[1]
            
    v1,m1 = res1
    v2,m2 = res2

    v = w1*v1 + w2*v2 + w1*w2*(outer(m1,m1) + outer(m2,m2) - outer(m1,m2) - outer(m2,m1))
    m = w1*m1 + w2*m2
    
    return v,m
    
    
    
def put_in_sample(sample,image_config_gen,m,task,ext_hash,sample_fs):
    out_record = SON([('model',m['config']['model']),
                      ('images',son_escape(image_config_gen['images'])),
                      ('task',son_escape(task)),
                 ])   

    sample = [t['filename'] for t in sample['train_data']]    
    filename = get_filename(out_record)
    out_record['filename'] = filename
    out_record['__hash__'] = ext_hash
    print('pickling split ...')
    out_data = cPickle.dumps(SON([('sample',sample)]))
    print('dump out sample ...')
    sample_fs.put(out_data,**out_record)

import bson           
def put_in_sample_result(res,image_config_gen,m,task,ext_hash,model_hash, image_hash, sampleres_fs):
    out_record = SON([('model',m['config']['model']),
                      ('model_hash',model_hash), 
                      ('images',son_escape(image_config_gen['images'])),
                      ('image_hash',image_hash),
                      ('task',son_escape(task))
                 ])   
                 
    filename = get_filename(out_record)
    out_record['filename'] = filename
    out_record['__hash__'] = ext_hash
    print('pickling sample result...')
    out_data = cPickle.dumps(SON([('sample_result',res)]))
    print('dumping out split result ...')
    sampleres_fs.put(out_data,**out_record)          


def prepare_corr(ext_hash,image_certificate_file,model_certificate_file,task):

    conn = pm.Connection(document_class=bson.SON)
    db = conn[DB_NAME]
    

    sample_coll = db['samples.files']
    sample_fs = gridfs.GridFS(db,'samples')
    remove_existing(sample_coll,sample_fs,ext_hash)
    extraction_coll = db['correlation_extraction.files']
    extraction_fs = gridfs.GridFS(db,'correlation_extraction')
    remove_existing(extraction_coll,extraction_fs,ext_hash)

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
    
    return model_configs,image_config_gen,model_hash,image_hash, task_list, sample_coll, sample_fs, extraction_coll, extraction_fs





############DATA SPLITTING#############
############DATA SPLITTING#############
############DATA SPLITTING#############
############DATA SPLITTING#############
############DATA SPLITTING#############

def generate_random_sample(task_config,hash,colname):

    ntrain = task_config['sample_size']
    ntest = 0
    N = 1

    query = task_config.get('query',{})
    cqueries = [reach_in('config',query)]
    
    return traintest.generate_multi_split2(DB_NAME,colname,cqueries,N,ntrain,ntest,universe={'__hash__':hash})[0]


def get_extraction_batches(image_hash,task,batch_size):
    if batch_size:
        conn = pm.Connection(document_class=bson.SON)
        db = conn[DB_NAME]
        coll = db['images.files']
        q = reach_in('config',task.get('query',SON([])))
        q['__hash__'] = image_hash
        count = coll.find(q).count()
        num_batches = int(math.ceil(count/batch_size))
        return [(batch_size*ind,batch_size*(ind+1)) for ind in range(num_batches)]
    else:
        return [None]
        
    
def get_extraction_configs(image_hash,task,batch):
    conn = pm.Connection(document_class=bson.SON)
    db = conn[DB_NAME]
    coll = db['images.files']
    q = reach_in('config',task.get('query',SON([])))
    q['__hash__'] = image_hash
    if batch:
        skip = batch[0]
        delta = batch[1] - skip 
        return coll.find(q,fields=['filename','config.image']).skip(skip).limit(delta)
    else:
        return coll.find(q,fields=['filename','config.image'])
        
def generate_splits(task_config,hash,colname,overlap=None):
    base_query = SON([('__hash__',hash)])
    ntrain = task_config['ntrain']
    ntest = task_config['ntest']
    N = task_config.get('N',10)
    base_query.update(reach_in('config',task_config.get('universe',SON([]))))
    
    query = task_config['query'] 
    if isinstance(query,list):
        cqueries = [reach_in('config',q) for q in query]
        return traintest.generate_multi_split2(DB_NAME,colname,cqueries,N,ntrain,
                                               ntest,universe=base_query,
                                               overlap=overlap)
    else:
        ntrain_pos = task_config.get('ntrain_pos')
        ntest_pos = task_config.get('ntest_pos')
        cquery = reach_in('config',query)
        return traintest.generate_split2(DB_NAME,colname,cquery,N,ntrain,ntest,
                                         ntrain_pos=ntrain_pos,ntest_pos = ntest_pos,
                                         universe=base_query,use_negate = True,
                                         overlap=overlap)


############TRANFORM AVERAGE#############
############TRANFORM AVERAGE#############
############TRANFORM AVERAGE#############
############TRANFORM AVERAGE#############
############TRANFORM AVERAGE#############

def transform_average(input,config,model_config):
    if isinstance(input,list):
        M = model_config['config']['model']
        assert isinstance(M,list) and len(M) == len(input)
        if isinstance(config,list):
            assert len(config) == len(M)
        else:
            config = [copy.deepcopy(config) for ind in range(len(M))]
        args = zip(input,config,[{'config':{'model':m}} for m in M])
        vec = sp.concatenate([transform_average(inp,conf,m) for (inp,conf,m) in args])
    else:
        vecs = []
        for level_input in input.values():
            K = level_input.keys()
            K.sort()
            vec = []
            if config:
                for cidx in K:
                    vec.append(average_transform(level_input[cidx],config,model_config))
            else:
                for cidx in K:
                    vec.append(unravel(level_input[cidx]))
            if vec[0].ndim > 0:
                vec = sp.concatenate(vec)
            vecs.append(vec)
        vec = sp.concatenate(vecs)
    
    return vec

def average_transform(input,config,M):
    if config['transform_name'] == 'translation':
        if config.get('max',False):
            V = [input.max(1).max(0)]
        elif config.get('various_stats',False):
            V = [max2d(input),min2d(input),mean2d(input),argmax2d(input),argmin2d(input)]
        else:
            V = [input.sum(1).sum(0)]
        if config.get('fourier',False):
            V = V + [np.abs(np.fft.fft(v)) for v in V]
        return sp.concatenate(V)
            
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
    elif config['transform_name'] == 'sum_up':
        model_config = M['config']['model']
        Lo = model_config['layers'][-2]
        L = model_config['layers'][-1]
        s1 = len(Lo['filter']['divfreqs'])
        s2 = Lo['filter']['norients']
        if L['filter']['model_name'] == 'uniform':
            os = L['filter'].get('osample',1)
            fs = L['filter'].get('fsample',1)
            s1 = s1/fs
            s2 = s2/os
        elif L['filter']['model_name'] == 'freq_uniform':
            os = L['filter'].get('osample',1)
            s2 = s2/os            
        return sum_up(input.max(1).max(0),s1,s2)
    elif config['transform_name'] == 'nothing':
        return input.ravel()
    elif config['transform_name'] == 'translation_and_fourier':
        return np.abs(np.fft.fft(input.sum(1).sum(0)))
    elif config['transform_name'] == 'central_slice':
        return get_central_slice(input,config['ker_shape'])
    else:
        raise ValueError, 'Transform ' + str(config['transform_name']) + ' not recognized.'



############ANALYSIS#############
############ANALYSIS#############
############ANALYSIS#############
############ANALYSIS#############


import scipy.stats as stats

def get_perfq(hash,q,p=75):
    conn = pm.Connection()
    coll = conn['thor']['performance']
    q['__hash__'] = hash
    res = np.array([l['test_accuracy'] for l in coll.find(q,fields=['test_accuracy'])])
    print(lof([res.max(),res.min(),res.mean(),stats.scoreatpercentile(res,p),res.std()]))
    return [res.max(),res.min(),res.mean(),stats.scoreatpercentile(res,p),res.std()]

def get_perf(hash,level,key,att,p=75):
    
    conn = pm.Connection()
    coll = conn['thor']['performance']

    level = str(level)
    q = {'__hash__':hash}
    vals = coll.find(q).distinct('model.layers.' + level + '.' + key + '.' + att)

    maxs = []
    mins = []
    means = []
    quartiles = []
    stds = []
    for val in vals:
        q['model.layers.' + level + '.' + key + '.' + att] = val
        res = np.array([l['test_accuracy'] for l in coll.find(q,fields=['test_accuracy'])])
        maxs.append(res.max())
        means.append(res.mean())
        mins.append(res.min())
        quartiles(stats.scoreatepercentile(res,p))
        stds.append(res.std())

    print('level %s, %s, %s' % (level,key,att))
    print('value: %s' % repr(vals))
    print('max: %s' % lof(maxs))
    print('mean: %s' % lof(means))
    print('quartiles %s' % lof(quartiles))
        
    return maxs,mins,means,quartiles,stds


def compute_multimodel_perfs(hash,level,key,att,num_models,p=75):

    conn = pm.Connection()
    coll = conn['thor']['performance']

    level = str(level)
    bq = {'__hash__':hash}
    vals = [coll.find(bq).distinct('model.' + str(mn) + '.layers.' + level + '.' + key + '.' + att) for mn in range(num_models)]
    sizes = tuple(map(len,vals))
    ranges = [range(s) for s in sizes]
  
    maxs = np.zeros(sizes)
    mins = np.zeros(sizes)
    means = np.zeros(sizes)
    quartiles = np.zeros(sizes)
    stds = np.zeros(sizes)
    for inds in itertools.product(*ranges):
        if all([inds[iv] < inds[iv+1] for iv in range(len(inds)-1)]):
            q = copy.deepcopy(bq)
            for (mn,ind) in zip(range(num_models),inds):
                q['model.' + str(mn) + '.layers.' + level + '.' + key + '.' + att] = vals[mn][ind]
            res = np.array([l['test_accuracy'] for l in coll.find(q,fields=['test_accuracy'])])
            if res:
                maxs[inds] = res.max()
                means[inds] = res.mean()
                mins[inds] = res.min()
                stds[inds] = res.std()
                quartiles[inds] = stats.scoreatpercentile(res,p)
        
    return maxs,mins,means,quartiles,stds

def compute_population_results(hash):
    conn = pm.Connection()
    db = conn[DB_NAME]
    coll = db['performance']
    L = list(coll.find({'__hash__':hash}))
    acc = np.array([l['test_accuracy'] for l in L])
    return acc.max(),acc.min(),acc.mean(),acc.std(),L[0]['task']


############CORE COMPUTATIONS###############
############CORE COMPUTATIONS###############
############CORE COMPUTATIONS###############
############CORE COMPUTATIONS###############
############CORE COMPUTATIONS###############
    
    
def compute_features(image_filename, image_fs, filter, model_config, convolve_func):
    image_fh = image_fs.get_version(image_filename)
    print('extracting', image_filename, model_config)
    return compute_features_core(image_fh,filter,model_config,convolve_func)    
    
    
def compute_features_core(image_fh,filters,model_config,convolve_func):
 
    m_config = model_config['config']['model']
    
    if isinstance(m_config,list):
        reslist = []
        for (filt,m) in zip(filters,m_config):
            image_fh.seek(0)
            res = compute_features_core(image_fh,filt,{'config':{'model':m}},convolve_func)
            reslist.append(res)
        return reslist
    else:
        conv_mode = m_config['conv_mode']    
        layers = m_config['layers']
        feed_up = m_config.get('feed_up',False)
        
        array = image2array(m_config,image_fh)
        array,orig_imga = preprocess(array,m_config)
        assert len(filters) == len(layers)
        dtype = array[0].dtype
        
        array_dict = {}
        for (ind,(filter,layer)) in enumerate(zip(filters,layers)):
            if feed_up:
                array_dict[ind-1] = array
        
            if filter is not None:
                array = fbcorr(array, filter, layer , convolve_func)
            
            if layer.get('lpool'):
                array = lpool(array,conv_mode,layer['lpool'])

            if layer.get('lnorm'):
                if layer['lnorm'].get('use_old',False):
                    array = old_norm(array,conv_mode,layer['lnorm'])
                else:
                    array = lnorm(array,conv_mode,layer['lnorm'])

        array_dict[len(layers)-1] = array
            
        return array_dict

def multiply(x,s1,s2,all=False,max=False,ravel=False):
    
    if ravel:
        x = x.ravel()
    elif max:
        x = x.max(1).max(0)
    else:
        x = x.sum(1).sum(0)
    
    if np.isnan(s1):
        y =  np.outer(x,x)[np.triu_indices(len(x))]
    else:
        if all:
            S = [[(s2*f1 + i , s2*f2 + (o + i) % s2)  for i in range(s2)] for f1 in range(s1) for f2 in range(s1) for o in range(s2) if f1 < f2 or (f1 == f2 and o != 0) ]
        else:
            S = [[(s2*f1 + i , s2*f1 + (o + i) % s2)  for i in range(s2)] for f1 in range(s1) for o in range(s2)]
        
        
        print(len(S),len(S[0]))
        y = np.array([sum([x[i]*x[j] for (i,j) in s]) for s in S])
    
    #y = y / np.sqrt((y**2).sum())
    #y = y / y.max()
    
    return y


def fbcorr(input,filter,layer_config,convolve_func):
    output = {}     
    for cidx in input.keys():
        if layer_config['filter']['model_name'] == 'multiply':
            (s1,s2) = filter
            output[cidx] = multiply(input[cidx],s1,s2,all=layer_config['filter'].get('all',False),
                                    max=layer_config['filter'].get('max',False),
                                    ravel=layer_config['filter'].get('ravel',False))
        else:
            min_out = layer_config['activ'].get('min_out')
            max_out=layer_config['activ'].get('max_out')
            if hasattr(min_out,'__iter__'):
                output[cidx] = convolve_func(input[cidx],
                                             filter,
                                             mode=layer_config['filter'].get('mode','valid'))                
                for ind  in range(output[cidx].shape[2]):
                    output[cidx][:,:,ind] = output[cidx][:,:,ind].clip(min_out[ind],max_out[ind])
            else:
                output[cidx] = convolve_func(input[cidx],
                                             filter,
                                             min_out=min_out,
                                             max_out=max_out,
                                             mode=layer_config['filter'].get('mode','valid'))         
    return output


def old_norm(input,conv_mode,params):
    output = {}
    for cidx in input.keys():
        if len(input[cidx].shape) == 3:
            inobj = input[cidx]
            strip = False
        else:
            strip = True
            inobj = input[cidx][:,:,sp.newaxis]

        if params:
    
            res = v1f.v1like_norm(inobj, conv_mode, params['inker_shape'],params['threshold'])
            if strip:
                res = res[:,:, 0]
            output[cidx] = res
        else: 
            output[cidx] = inobj
            
    return output


def lpool(input,conv_mode,config):
    pooled = {}
    for cidx in input.keys():
        pooled[cidx] = pythor3.operation.lpool(input[cidx],plugin='cthor',**config)
    return pooled

        
 
def lnorm(input,conv_mode,config):
    normed = {}
    if 'inker_shape' in config: 
        config['inker_shape'] = tuple(config['inker_shape'])
    if 'outker_shape' in config:
        config['outker_shape'] = tuple(config['outker_shape'])
    for cidx in input.keys():
        normed[cidx] = pythor3.operation.lnorm(input[cidx],plugin='cthor',**config)
    return normed

def c_numpy_mixed(arr_in, arr_fb, arr_out=None,
                 mode=DEFAULT_MODE,
                 min_out=DEFAULT_MIN_OUT,
                 max_out=DEFAULT_MAX_OUT,
                 stride=DEFAULT_STRIDE
                ):
    
    if max(arr_fb.shape[-3:-1]) > 19:
        return pythor3.operation.fbcorr(arr_in, arr_fb, arr_out=None,
                 mode=mode,
                 min_out=min_out,
                 max_out=max_out,
                 stride=stride,
                 plugin="numpyFFT",
                 plugin_kwargs={"use_cache":True})
    else:
        return pythor3.operation.fbcorr(arr_in, arr_fb, arr_out=None,
                 mode=mode,
                 min_out=min_out,
                 max_out=max_out,
                 stride=stride,
                 plugin="cthor")


     
###########UTILS#########
###########UTILS#########
###########UTILS#########
###########UTILS#########
###########UTILS#########


def get_from_cache(obj,cache):
    hash = hashlib.sha1(repr(obj)).hexdigest()
    if hash in cache:
        print('using cache for %s' % str(hash))
        return cache[hash]
        
def put_in_cache(obj,value,cache):
    hash = hashlib.sha1(repr(obj)).hexdigest()
    cache[hash] = value
     

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

def lof(mylist):
    return ", ".join(['%0.2f' % item for item in mylist])


def get_data_batches(data,num_batches):

    bs = int(math.ceil(float(len(data)) / num_batches))
    
    return [data[bs*i:bs*(i+1)] for i in range(num_batches)]

def remove_existing(coll,fs, hash):
    existing = coll.find({'__hash__':hash})
    for e in existing:
        fs.delete(e['_id'])


def get_config(config_fname):
    config_path = os.path.abspath(config_fname)
    print("Config file:", config_path)
    config = {}
    execfile(config_path, {},config)
    
    return config['config']
    
def unravel(X):
    return sp.concatenate([X[:,:,i].ravel() for i in range(X.shape[2])])
        
def sum_up(x,s1,s2):
    y = []
    F = lambda x : (s2**2)*x[2] + s2*x[0] + x[1]
    for i in range(s1):
        S = [map(F,[(k,(j+k)%s2,i) for k in range(s2)]) for j in range(s2)]
        y.extend([sum([x[ss] for ss in s]) for s in S])
    return np.array(y)
    
def mean2d(x):
    if x.ndim <= 2:
        return np.array([x.mean()])
    else:
        return np.array([x[:,:,i].mean() for i in range(x.shape[2])])

def max2d(x):
    if x.ndim <= 2:
        return np.array([x.max()])
    else:
        return np.array([x[:,:,i].max() for i in range(x.shape[2])])
    
def min2d(x):
    if x.ndim <= 2:
        return np.array([x.min()])
    else:
        return np.array([x[:,:,i].min() for i in range(x.shape[2])])

def argmax2d(x):
    if x.ndim <= 2:
        return np.array([x.argmax()])
    else:
        return np.array([x[:,:,i].argmax() for i in range(x.shape[2])])

def argmin2d(x):
    if x.ndim <= 2:
        return np.array([x.argmin()])
    else:
        return np.array([x[:,:,i].argmin() for i in range(x.shape[2])])


