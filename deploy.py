import os

import starflow.de as de
from starflow.protocols import actualize, activate
from v1like_extract import get_config 
 
import pythor_protocols as protocols


def get_code_dir(hash):
    manager = de.DataEnvironmentManager() 
    working_de = manager.working_de
    return os.path.join(working_de.relative_generated_code_dir,protocols.DB_NAME.replace('-','_'),'module_' +hash)


@activate(lambda x : x[0], lambda x: get_code_dir(protocols.image_protocol_hash(x[0])))
def images(config_path,parallel=False,reads=None):
    D,hash = protocols.image_protocol(config_path,write = False,parallel=parallel,reads=reads)
    actualize(D,outfiledir=get_code_dir(hash))
    return hash

@activate(lambda x : x[0], lambda x: get_code_dir(protocols.model_protocol_hash(x[0])))
def models(config_path,parallel=False):
    D,hash = protocols.model_protocol(config_path,write = False,parallel=parallel)
    actualize(D,outfiledir=get_code_dir(hash))
    return hash
    
    
def extract_features(image_config_path,
                                   model_config_path,
                                   convolve_func_name = 'numpy',
                                   parallel=False,
                                   batch_size=1000):
    D,hash = protocols.extract_features_protocol(image_config_path,
                            model_config_path,
                            convolve_func_name = convolve_func_name,
                            write = False,
                            parallel=parallel,
                            batch_size=batch_size)
    actualize(D,outfiledir=get_code_dir(hash))
    return hash
    

 
def evaluate(evaluate_config_path,model_config_path,image_config_path):
    D,hashes = protocols.evaluate_protocol(evaluate_config_path,model_config_path,image_config_path,write=False)
    for (d,h) in zip(D,hashes):
        actualize([d],outfiledir=get_code_dir(h))
    return hashes

@activate(lambda x : (x[0],x[1],x[2]), lambda x: tuple([get_code_dir(x) for x in protocols.get_extract_and_evaluate_hashes(x[0],x[1],x[2])]))
def extract_and_evaluate(evaluate_config_path,
                                       model_config_path,
                                       image_config_path,
                                       convolve_func_name='numpy'):
    DH = protocols.extract_and_evaluate_protocol(evaluate_config_path,model_config_path,
                                             image_config_path,
                                             convolve_func_name=convolve_func_name,
                                             write=False)
                                             
    for (h,ops) in DH.items():
        actualize(ops,outfiledir=get_code_dir(h))
    
    return DH.keys()


def run_full_protocol(config_path,
                      extract_and_evaluate=True,
                      parallel=False,
                      feature_batch_size=1000,
                      convolve_func_name='numpy'):
    
    config = get_config(config_path)
    hashes = []
    if 'images' in config:
        im_hash = images(config_path,parallel=parallel)
        hashes.append(im_hash)
    if 'models' in config:
        m_hash = models(config_path,parallel=parallel)
        hashes.append(m_hash)
    if 'images' in config and 'models' in config:
        if not 'train_test' in config:
            f_hash = extract_features(config_path,
                             config_path,
                             convolve_func_name=convolve_func_name,
                             parallel=parallel,
                             batch_size=feature_batch_size)
            hashes.append(f_hash)
                             
        elif extract_and_evaluate:
            e_hashes = extract_and_evaluate(config_path,config_path,config_path,
                                 convolve_func_name=convolve_func_name)
            hashes.extend(e_hashes)
        else:
            f_hash = extract_features(config_path,
                             config_path,
                             convolve_func_name=convolve_func_name,
                             parallel=parallel,
                             batch_size=feature_batch_size)
            e_hashes = evaluate(config_path,config_path,config_path)
            hashes.append(f_hash).extend(e_hashes)

    codes = ['^.*' + h for h in hashes]
    update.FullUpdate(AU=codes)
    
    
        
