def test_protocol(test_evaluation_config_path,train_evaluation_config_path,test_extraction_config_path,train_extraction_config_path,model_config_path,test_image_config_path,train_image_config_path,write=False,parallel=False,use_db=False):
                        
    model_config_gen = get_config(model_config_path)
    model_config_gen = model_config_gen.pop('models')
    model_hash = get_config_string(model_config_gen)
    model_certificate = '../.model_certificates/' + model_hash
    
    test_image_config_gen = get_config(test_image_config_path)
    test_image_config_gen = test_image_config_gen.pop('images')
    test_image_hash =  get_config_string(test_image_config_gen)
    test_image_certificate = '../.image_certificates/' + test_image_hash

    train_image_config_gen = get_config(train_image_config_path)
    train_image_config_gen = train_image_config_gen.pop('images')
    train_image_hash =  get_config_string(train_image_config_gen)
    train_image_certificate = '../.image_certificates/' + train_image_hash

    test_extraction_config = get_config(test_extraction_config_path)
    test_extraction_config = test_extraction_config.pop('extractions')
    
    train_extraction_config = get_config(train_extraction_config_path)
    train_extraction_config = train_extraction_config.pop('extractions')    

    train_evaluation_config = get_config(train_evaluation_config_path)
    train_evaluation_config = evaluation_config.pop('train_test')

    test_evaluation_config = get_config(test_evaluation_config_path)
    test_evaluation_config = evaluation_config.pop('test')

    D = []
    DH = {}
    
      
    for test_extraction in test_extraction_config:
        test_extraction_config_gen = SON([('models',model_config_gen),('images',test_image_config_gen),('extraction',test_extraction)])
        test_extraction_hash = get_config_string(test_extraction_config_gen)
        test_extraction_certificate = '../.extraction_certificates/' + test_extraction_hash
        
        for train_extraction in train_extraction_config:
            train_extraction_config_gen = SON([('models',model_config_gen),('images',train_image_config_gen),('extraction',train_extraction)])
            train_extraction_hash = get_config_string(train_extraction_config_gen)
            train_extraction_certificate = '../.extraction_certificates/' + train_extraction_hash        
            
            for train_evaluation in train_evaluation_config:
                train_evaluation_config_gen = SON([('models',model_config_gen),('images',train_image_config_gen),('extraction',train_extraction),('train_test',train_evalation_config)])
                train_evaluation_hash = get_config_string(train_evaluation_config_gen)
                train_evaluation_certificate = '../.performance_certificates/' + train_evaluation_hash    
                
                for test_evaluation in test_evaluation_config:
                    
                    overall_config_gen = SON([('models',model_config_gen),
                                              ('test_images',test_image_config_gen),
                                              ('train_images',train_image_config_gen),
                                              ('test_extraction',test_extraction),
                                              ('train_extraction',train_extraction),
                                              ('test_evaluation',test_evalation),
                                              ('train_evaluation',train_evaluation)])
                                              
                    test_evaluation_hash = get_config_string(overall_config_gen)    
                    
                    test_evaluation_certificate = '../.prediction_certificates/' + test_evaluation_hash
                                                             
                    op = ('test_' + test_hash,test_func,  (test_evaluation_certificate,
                                                      train_evaluation_certificate,
                                                      test_extraction_certificate, 
                                                      train_extraction_certificate
                                                      test_image_certificate,
                                                      train_image_certificate,
                                                      model_certificate,
                                                      test_evaluation,
                                                      test_evaluation_hash,
                                                      train_evaluation,
                                                      train_evaluation_hash,
                                                      test_extraction,
                                                      test_extraction_hash,
                                                      train_extraction,
                                                      train_extraction_hash,
                                                      use_db))                                                
                    D.append(op)
                    DH[evaluation_hash] = [op]
                 
    if write:
        actualize(D)
    return DH
    
    

@activate(lambda x : (x[1],x[2],x[3],x[4],x[5],x[6]),lambda x : x[0])
def test_func( test_evaluation_certificate,
          train_evaluation_certificate,
          test_extraction_certificate, 
          train_extraction_certificate
          test_image_certificate,
          train_image_certificate,
          model_certificate,
          test_evaluation,
          test_evaluation_hash,
          train_evaluation,
          train_evaluation_hash,
          test_extraction,
          test_extraction_hash,
          train_extraction,
          train_extraction_hash,
          use_db):

    
    conn = pm.Connection(document_class=bson.SON)
    db = conn[DB_NAME]
    predictions_col = db['predictions']

    model_configs,train_extraction_task_list,train_evaluation_task_list,
    test_extraction_task_list,test_evaluation_task_list,
    splitperf_col,feature_col,feature_fs,predictions_col = prepare_test(test_evaluation_hash,model_certificate_file,test_image_certificate_file,
                                                                        train_image_certificate_file,train_extraction,test_extraction,train_evaluation,test_evaluation)
                                                
    for m in model_configs: 
        print('Testing model',m)

        for train_extraction_task in train_extraction_task_list:
            for train_evaluation_task in train_evaluation_task_list:  
                split_results = get_most_recent_files(splitperf_coll,{'__hash__':train_evaluation_hash,
                                                                      'task':son_escape(train_evaluation_task),
                                                                      'extraction':son_escape(train_extraction_task),
                                                                      'model':m['config']['model']})     
                                                                      
                for test_extraction_task in test_extraction_task_list:
                    test_data = get_most_recent_files(feature_col,{'__hash__':test_extraction_hash,
                                                                   'extraction':son_escape(test_extraction_task),
                                                                   'model':m['config']['model'])
                    test_filenames = [t['filename'] for t in test_data]
                    for test_evaluation_task in test_evaluation_task_list:
                        test_features = load_features_batch(test_filenames,
                                                            feature_col,
                                                            feature_fs,
                                                            m,
                                                            test_evaluation_task,
                                                            test_extraction,
                                                            test_extraction_hash,
                                                            use_db)
                        for (split_num,split) in enumerate(splits):
                            cls_data = split['cls_data']
                            weights = cls_data['coef']
                            bias = cls_data['intercept']
                            labels = cls_data['labels']
                            test_margins = sp.dot(test_features,weights) + bias
                            test_predictions = labels[test_margins.argmax(1)]
                            for (test_filename,test_margin,test_prediction) in zip(test_filenames,test_margins,test_predictions):
                                put_in_predictions(test_filename,
                                                         list(test_margin),
                                                         test_prediction,
                                                         split_num,
                                                         labels,
                                                         m,
                                                         model_hash,
                                                         test_image_hash,
                                                         train_image_hash,
                                                         test_extraction_hash,
                                                         train_extraction_hash,
                                                         test_evaluation_hash,
                                                         train_evaluation_hash,
                                                         test_evaluation_task,              
                                                         test_extraction_task,
                                                         train_evalution_task,
                                                         train_extraction_task,
                                                         predictions_col)    
                                
                        
                        
                        
def put_in_predictions(test_filename,
                         test_margin,
                         test_prediction,
                         split_num,
                         labels,
                         m,
                         model_hash,
                         test_image_hash,
                         train_image_hash,
                         test_extraction_hash,
                         train_extraction_hash,
                         test_evaluation_hash,
                         train_evaluation_hash,
                         test_evaluation_task,              
                         test_extraction_task,
                         train_evalution_task,
                         train_extraction_task,
                         predictions_col):
    

    out_record = SON([('image_filename',test_filename),
    ('split_num',split_num),
    ('model',m['config']['model']),
    ('model_filename',m['filename']),
    ('model_hash',model_hash),
    ('test_image_hash',test_image_hash),
    ('train_image_hash',train_image_hash),
    ('test_extraction_hash',test_extraction_hash),
    ('train_extraction_hash',train_extraction_hash),
    ('train_evaluation_hash',train_evaluation_hash)
    ('__hash__',test_evaluation_hash),
    ('test_evaluation_task',son_escape(test_evaluation_task)),
    ('train_evaluation_task',son_escape(train_evaluation_task)),
    ('test_extraction_task',son_escape(test_extraction_task)),
    ('train_extraction_task',son_escape(train_extraction_task)),
    ('margins',margin),
    ('prediction',prediction),  
    ('labels',labels)])

    print('inserting result ...')
    predictions_coll.insert(out_record)
                        
                        
                        
def prepare_test(test_evaluation_hash,model_certificate_file,test_image_certificate_file,train_image_certificate_file,train_extraction,test_extraction,train_evaluation,test_evaluation)

    conn = pm.Connection(document_class=bson.SON)
    db = conn[DB_NAME]

    print('preparing predictions ...')
    predictions_coll = db['predictions']
    predictions_coll.remove({'__hash__':test_evaluation_hash})
    
    print('preparing splitperfs and features...')
    splitperf_coll = db['split_performance.files']
    feature_col = db['features.files']
    feature_fs = gridfs.GridFS(db,'features')
    
    print('preparing models ...')
    model_certdict = cPickle.load(open(model_certificate_file))
    model_hash = model_certdict['model_hash']
    model_coll = db['models.files']
    mquery = {'__hash__' : model_hash}
    model_configs = get_most_recent_files(model_coll,mquery)

    print('preparing testimages ...')
    test_image_certdict = cPickle.load(open(test_image_certificate_file))
    test_image_hash = test_image_certdict['image_hash']
    test_image_config_gen = test_image_certdict['args']
    
    print('preparing train images ...')
    train_image_certdict = cPickle.load(open(train_image_certificate_file))
    train_image_hash = train_image_certdict['image_hash']
    train_image_config_gen = train_image_certdict['args']    
    
    
    if isinstance(train_extraction,list):
        train_extraction_task_list = train_extraction
    else:
        train_extraction_task_list = [train_extraction]
    
    if isinstance(test_extraction,list):
        test_extraction_task_list = test_extraction
    else:
        test_extraction_task_list = [test_extraction]
        
    if isinstance(train_evaluation,list):
        train_evaluation_task_list = train_evaluation
    else:
        train_evaluation_task_list = [train_evaluation]
        
    if isinstance(test_evaluation,list):
        test_evaluation_task_list = test_evaluation
    else:
        test_evaluation_task_list = [test_evaluation]        
        
    return model_configs,train_extraction_task_list,train_evaluation_task_list,test_extraction_task_list,test_evaluation_task_list,splitperf_col,feature_col,feature_fs,predictions_col
                         
