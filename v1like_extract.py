import sys
import os
import time
import os
import hashlib
import cPickle

from collections import OrderedDict

import numpy as np
import scipy as sp
from starflow.protocols import protocolize, actualize

import v1like_funcs as v1f 
import filter_generation
from dbutils import dot,cross,inject, DBAdd
from processing import image2array, preprocess, postprocess
from rendering import renderman_config_gen, cairo_config_gen, renderman_render, cairo_render

#=-=-=-=-=-=-=-=-=-=-=-=-=
#determine GPU support
#=-=-=-=-=-=-=-=-=-=-=-=-=
try:
    import v1_pyfft
except:
    GPU_SUPPORT = False
else:
    GPU_SUPPORT = True



#=-=-=-=-=-=-=-=-=-=-=-=-=
#=-=-=-=-=-=-=-=-=-=-=-=-=
#the main protocol
#=-=-=-=-=-=-=-=-=-=-=-=-=
#=-=-=-=-=-=-=-=-=-=-=-=-=

def v1_feature_extraction_protocol(config_path,use_cpu = False):
    D = DBAdd(v1_initialize,args = (config_path,use_cpu))
    actualize(D)


def v1_initialize(config_path,use_cpu):
    if use_cpu or not GPU_SUPPORT:    
        extract_func = numpy_extract
    else:
        extract_func = pyfft_extract

    config = get_config(config_path)
    
    image_params = OrderedDict([('image',config['image'])])
    model_params = OrderedDict([('model',config['model'])])

    if model_params['model']['filter']['model_name'] in ['random','random_gabor']:
        model_params['model']['filter']['id'] = random_id()
    
    return [('generate_images', render_image, {'args':(image_params,)}),                         
            ('generate_models', get_filterbank, {'args':(model_params,)}),            
            ('extract_features',extract_func)]



#=-=-=-=-=-=-=-=-=-=-=-=-=
#=-=-=-=-=-=-=-=-=-=-=-=-=
#the DB operations
#=-=-=-=-=-=-=-=-=-=-=-=-=
#=-=-=-=-=-=-=-=-=-=-=-=-=

######image generation
def image_config_gen(image_params):
    args = image_params['image']
    generator = args['generator']
    if generator == 'renderman':
        params = renderman_config_gen(args)
    elif generator == 'cairo':
        params = cairo_config_gen(args)
    else:
        raise ValueError, 'image generator not recognized'     
    for p in params:
        p['image']['generator'] = generator
    return params
    
@inject('v1','raw_images',image_config_gen)
def render_image(config): 
     config = config['image'].copy()
     generator = config.pop('generator')
     if generator == 'renderman':
         return renderman_render(config)
     elif generator == 'cairo':
         return cairo_render(config)
     else:
         raise ValueError, 'image generator not recognized'
   
   
######filter generation   
def model_config_generator(query): return [query]
    
@inject('v1','models', model_config_generator)
def get_filterbank(config):
    filterbank = filter_generation.get_filterbank(config['model'])    
    return cPickle.dumps(filterbank)
    
      
######extraction
def extract_features(fhs,config,convolve_func):

    image_config = config[0]
    model_config = config[1]['model']
    image_fh = fhs[0] 
    filter_fh = fhs[1]
    
    conv_mode = model_config['conv_mode']
    
    #preprocessing
    array = image2array(model_config,image_fh)
    preprocessed,orig_imga = preprocess(array,model_config)
    
    #input normalization
    norm_in = norm(preprocessed,conv_mode,model_config['normin'])
    
    #filtering
    filtered = convolve(norm_in, filter_fh, model_config, convolve_func)

    #nonlinear activation
    activ = activate(filtered,model_config)
    
    #output normalization
    norm_out = norm(activ,conv_mode,model_config['normout'])
    
    #pooling
    pooled = pool(norm_out,conv_mode,model_config)
        
    #postprocessing
    fvector_l = postprocess(norm_in,filtered,activ,norm_out,pooled,orig_imga,model_config['featsel']) 

    output = sp.concatenate(fvector_l).ravel()
    
    return cPickle.dumps(output)


@cross('v1',['raw_images','models'],'features')
def numpy_extract(fhs,config):
    return extract_features(fhs,config,v1f.v1like_filter_numpy)
    
if GPU_SUPPORT:    
    @cross('v1',['raw_images','models'],'features',setup = v1_pyfft.setup_pyfft,cleanup = v1_pyfft.cleanup_pyfft)
    def pyfft_extract(fhs,config):
        return extract_features(fhs, config, v1f.v1like_filter_pyfft)



#=-=-=-=-=-=-=-=-=-=
#=-=-=-=-=-=-=-=-=-=
#random utilities
#=-=-=-=-=-=-=-=-=-=
#=-=-=-=-=-=-=-=-=-=

def pool(input,conv_mode,config):
    pooled = {}
    for cidx in input.keys():
        pooled[cidx] = v1f.v1like_pool(input[cidx],conv_mode,**config['pool'])
    return pooled

def activate(input,config):
    minout = config['activ']['minout'] # sustain activity
    maxout = config['activ']['maxout'] # saturation
    activ = {}
    for cidx in input.keys():
       activ[cidx] = input[cidx].clip(minout, maxout)
    return activ

def norm(input,conv_mode,params):
    output = {}
    for cidx in input.keys():
       if len(input[cidx].shape) == 3:
          inobj = input[cidx]
       else:
          inobj = input[cidx][:,:,sp.newaxis]
       output[cidx] = v1f.v1like_norm(inobj, conv_mode, **params)
    return output

def convolve(image,filter_fh,model_config,convolve_func):
    def filter_source():
        filter_fh.seek(0)
        return cPickle.loads(filter_fh.read())
    output = {}
    for cidx in image.keys():
        output[cidx] = convolve_func(image[cidx][:,:,0],filter_source,model_config)
    return output

def random_id():
    hashlib.sha1(str(np.random.randint(10,size=(32,)))).hexdigest()

def get_config(config_fname):
    config_path = os.path.abspath(config_fname)
    print("Config file:", config_path)
    config = {}
    execfile(config_path, {},config)
    
    return config['config']


    



