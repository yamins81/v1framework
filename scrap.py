def extract_random_protocol(extract_config_path,model_config_path,image_config_path,convolve_func_name='numpy', write=False,parallel=False):
    model_config_gen = get_config(model_config_path)
    model_hash = get_config_string(model_config_gen['models'])
    model_certificate = '../.model_certificates/' + model_hash
    
    image_config_gen = get_config(image_config_path)
    image_hash =  get_config_string(image_config_gen['images'])
    image_certificate = '../.image_certificates/' + image_hash

    extract_config = get_config(extract_config_path)
    task_config = extract_config.pop('extractions')

    D = []
    DH = {}
    for task in task_config:
        overall_config_gen = SON([('models',model_config_gen),('images',image_config_gen),('task',task)])
        ext_hash = get_config_string(overall_config_gen)    
        
        performance_certificate = '../.extraction_certificates/' + ext_hash
        if not parallel:
            op = ('extraction_' + ext_hash,extract_random,(performance_certificate,image_certificate,model_certificate,extract_config_path,convolve_func_name,task,ext_hash))
        else:
            op = ('extraction_' + ext_hash,extract_random_parallel,(performance_certificate,image_certificate,model_certificate,evaluate_config_path,convolve_func_name,task,ext_hash))
        D.append(op)
        DH[ext_hash] = [op]
             
    if write:
        actualize(D)
    return DH
    

@activate(lambda x : (x[1],x[2],x[3]),lambda x : x[0])
def extract_random_parallel(outfile,image_certificate_file,model_certificate_file,cpath,convolve_func_name,task,ext_hash):
        
    (model_configs, image_config_gen, model_hash, image_hash, task_list,
     sample_coll, sample_fs, extraction_coll, extraction_fs) = prepare_extract_random(ext_hash,
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
            put_in_sample(sample,image_config_gen,m,task,ext_hash,ind,sample_fs)  
            jobid = qsub(extract_random_parallel_core,(image_config_gen,m,task,ext_hash,convolve_func_name),opstring=opstring)
            print('Submitted job', jobid)
            jobids.append(jobid)
                
    print('Waiting for jobs', jobids) 
    statuses = wait_and_get_statuses(jobids)
    
    if not all([status == 0 for status in statuses]):
        bad_jobs = [jobid for (jobid,status) in zip(jobids,statuses) if not status == 0]
        raise ValueError, 'There was a error in job(s): ' + repr(bad_jobs)
    
    createCertificateDict(outfile,{'image_file':image_certificate_file,'models_file':model_certificate_file})
    
def extract_random_parallel_core(image_config_gen,m,task,ext_hash,convolve_func_name,cache_port=None):

    if cache_port is None:
        cache_port = NETWORK_CACHE_PORT
    cache_port = None
        

               
    conn = pm.Connection(document_class=bson.SON)
    db = conn[DB_NAME]
    sample_col = db['samples.files']
    sample_fs = gridfs.GridFS(db,'samples')

    sampleconf = get_most_recent_files(sample_col,{'__hash__':ext_hash,'model':m['config']['model'],'images':son_escape(image_config_gen['images'])})[0]
    sample = cPickle.loads(sample_fs.get_version(sampleconf['filename']).read())['sample']
    res = extract_random_core(sample,m,convolve_func_name,task,cache_port)
    extraction_fs = gridfs.GridFS(db,'sample_extraction')
    put_in_sample_result(res,image_config_gen,m,task,ext_hash,extraction_fs)



def extract_random_core(sample,m,convolve_func_name,task,cache_port):

    sample_filenames = [t['filename'] for t in sample]

    existing_extractions = [get_from_cache((tf,m,task.get('transform_average')),FEATURE_CACHE) for tf in sample]
    new_sample_filenames = [sample_filenames[i] for (i,x) in enumerate(existing_extractions) if x is None]

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
        batches = get_data_batches(new_sample_filenames,num_batches)
        results = []
        for (bn,b) in enumerate(batches):
            results.append(pool.apply_async(extract_random_inner_core,(b,m.to_dict(),convolve_func_name,bn,task.to_dict(),cache_port)))
        results = [r.get() for r in results]
        new_extractions = ListUnion(results)
    else:
        new_extractions = extract_and_evaluate_inner_core(new_sample_filenames,m,convolve_func_name,0,task,cache_port)

    #TODO get the order consistent with original ordering
    extractions = filter(lambda x : x is not None,existing_train_features) + new_extractions
     
    for (im,f) in zip(new_sample_filenames,new_extractions):
        put_in_cache((im,m,task.get('transform_average')),f,FEATURE_CACHE)
          
    
    return extractions
    


@activate(lambda x : (x[1],x[2],x[3]),lambda x : x[0])
def extract_random(outfile,image_certificate_file,model_certificate_file,cpath,convolve_func_name,task,ext_hash):

    (model_configs, image_config_gen, model_hash, image_hash, task_list, 
     sample_coll, sample_fs,extraction_coll,extraction_fs) = prepare_extract_random(ext_hash,
                                                image_certificate_file,
                                                model_certificate_file,
                                                task)
    for m in model_configs: 
        print('Evaluating model',m)
        for task in task_list:  
            print('task',task)
            sample = generate_random_sample(task,image_hash,'images') 
            put_in_sample(sample,image_config_gen,m,task,ext_hash,ind,sample_fs)  
            res = extract_random_core(sample,m,convolve_func_name,task,None)    
            put_in_sample_result(res,image_config_gen,m,task,ext_hash,extraction_fs)
    
    createCertificateDict(outfile,{'image_file':image_certificate_file,'models_file':model_certificate_file})
    
    
    
def put_in_sample(sample,image_config_gen,m,task,ext_hash,sample_fs):
    pass

import bson           
def put_in_sample_result(res,image_config_gen,m,task,ext_hash,extraction_fs):
    pass



def prepare_extract_random(ext_hash,image_certificate_file,model_certificate_file,task):

    conn = pm.Connection(document_class=bson.SON)
    db = conn[DB_NAME]
    

    sample_coll = db['samples.files']
    sample_fs = gridfs.GridFS(db,'samples')
    remove_existing(sample_coll,sample_fs,ext_hash)
    extraction_coll = db['sample_extraction.files']
    extraction_fs = gridfs.GridFS(db,'sample_extraction')
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
