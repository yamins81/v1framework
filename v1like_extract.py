import sys
import os
import time
import os
import hashlib
import cPickle

#from collections import OrderedDict
from bson import SON

import numpy as np
import scipy as sp
from starflow.protocols import protocolize, actualize
from starflow.utils import is_string_like

import v1like_funcs as v1f 
import filter_generation
from dbutils import dot,cross,inject, aggregate, DBAdd
from processing import image2array, preprocess, postprocess
from rendering import renderman_config_gen, cairo_config_gen, renderman_render, cairo_render, cairo_random_config_gen

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
    

def v1_feature_extraction_protocol(config_path,use_cpu = False,write=False):
    D = DBAdd(v1_initialize,args = (config_path,use_cpu))
    if write:
        actualize(D)
    return D

def v1_initialize(config_path,use_cpu):
    if use_cpu or not GPU_SUPPORT:    
        extract_func = numpy_extract
    else:
        extract_func = pyfft_extract

    config = get_config(config_path)
    
    image_params = SON([('image',config['image'])])
    models_params = config['models']
    for model_params in models_params:
        if model_params['filter']['model_name'] in ['really_random','random_gabor']:
            model_params['id'] = config_path
    
    return [{'step':'generate_images','func':render_image, 'params':(image_params,)},                         
            {'step':'generate_models', 'func':get_filterbank,'params':(models_params,)},            
            {'step':'extract_features','func':extract_func}]



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


def random_image_config_gen(image_params):
    args = image_params['image']
    generator = args['generator']
    if generator == 'renderman':
        params = renderman_random_config_gen(args)
    elif generator == 'cairo':
        params = cairo_random_config_gen(args)
    else:
        raise ValueError, 'image generator not recognized'     
    for p in params:
        p['image']['generator'] = generator
    return params   
    
@inject('v1','image_configs',random_image_config_gen)
def generate_random_image_configs(config): 
     return config
   
   
######filter generation   
def model_config_generator(models): return [SON([('model',m)]) for m in models]
    
@inject('v1','models', model_config_generator)
def get_filterbank(config):
    result = filter_generation.get_filterbank(config['model'])
    if isinstance(result,dict):
        assert 'filterbank' in result
        result['__file__'] = cPickle.dumps(result.pop('filterbank'))
        return result
    else:
        return cPickle.dumps(result)
    
      
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
    norm_in = norm(preprocessed,conv_mode,model_config.get('normin'))
    
    #filtering
    filtered = convolve(norm_in, filter_fh, model_config, convolve_func)

    #nonlinear activation
    activ = activate(filtered,model_config.get('activ'))
    
    #output normalization
    norm_out = norm(activ,conv_mode,model_config.get('normout'))
    #pooling
    pooled = pool(norm_out,conv_mode,model_config.get('pool'))
        
    #postprocessing
    fvector_l = postprocess(norm_in,filtered,activ,norm_out,pooled,orig_imga,model_config.get('featsel'))

    output = sp.concatenate(fvector_l).ravel()
    
    return cPickle.dumps(output)


@cross('v1',['raw_images','models'],'features')
def numpy_extract(fhs,config):
    return extract_features(fhs,config,v1f.v1like_filter_numpy)
    
if GPU_SUPPORT:    
    @cross('v1',['raw_images','models'],'features',setup = v1_pyfft.setup_pyfft,cleanup = v1_pyfft.cleanup_pyfft)
    def pyfft_extract(fhs,config):
        return extract_features(fhs, config, v1f.v1like_filter_pyfft)


#=-=-=-=-=-=-=-=-=-=-=
#=-=-=-=-=-=-=-=-=-=-=
#experimental learning
#=-=-=-=-=-=-=-=-=-=-=
#=-=-=-=-=-=-=-=-=-=-=

@aggregate('v1','raw_images','raw_images','test_learned_models')
def learn_filterbank(fhs,configs):
    fhs = [fh[0] for fh in fhs]

    config = configs[0][0].copy()
    
    num_slices = config['model']['num_slices']
    filterbank = filter_generation.get_filterbank(config['model'])
    
    fh,fw,num_filters = filterbank.shape
    filter_kshape = (fh,fw)
    
    counts = np.zeros(num_filters)
    
    for fh in fhs:
        array = image2array(config['model'],fh)[:,:,0]
     
        slices = [get_random_slice(array,filter_kshape) for i in range(num_slices)]
        for s in slices:
            patch = array[s]
            distarray = []
            for i in range(num_filters):
                d = dist(filterbank[:,:,i],patch)
                distarray.append(d)
            distarray = np.array(distarray)
            imax = distarray.argmax()
            counts[imax] += 1
            lr = .0001
            #lr = 1./counts[imax]
            filterbank[:,:,imax] = normalize(filterbank[:,:,imax]*(1 - lr) + patch*lr)
            
            
    return cPickle.dumps(filterbank)
    
def dist(x,y):
    return np.sqrt(((x - y)**2).sum())
   
def get_initial_filterbank(config):
    return get_filterbank(config)
    
def get_random_slice(array,shape):
    ashape = array.shape
    shapediff = [max(0,a-s) for (a,s) in zip(ashape,shape)]
    rands = [np.random.randint(0,high = d+1) for d in shapediff] 
    s = tuple([slice(r,r+s) for (r,s) in zip(rands,shape)])
    return s

def normalize(Y):
    m = Y.mean()
    return (Y - m) / np.sqrt(((Y - m)**2).sum())

#=-=-=-=-=-=-=-=-=-=
#=-=-=-=-=-=-=-=-=-=
#random utilities
#=-=-=-=-=-=-=-=-=-=
#=-=-=-=-=-=-=-=-=-=

def pool(input,conv_mode,config):
    if config:
        pooled = {}
        for cidx in input.keys():
            pooled[cidx] = v1f.v1like_pool(input[cidx],conv_mode,**config)
        return pooled
    return input
        
def activate(input,config):
    if config:
        minout = config['minout'] # sustain activity
        maxout = config['maxout'] # saturation
        activ = {}
        for cidx in input.keys():
            #activ[cidx] = input[cidx].clip(minout, maxout) 
            activ[cidx] = (input[cidx] - minout).clip(0, maxout-minout)
        return activ 
    else:
        return input
        

def norm(input,conv_mode,params):
    output = {}
    for cidx in input.keys():
        if len(input[cidx].shape) == 3:
            inobj = input[cidx]
        else:
            inobj = input[cidx][:,:,sp.newaxis]
        if params:
            output[cidx] = v1f.v1like_norm(inobj, conv_mode, **params)
        else: 
            output[cidx] = inobj
    return output


def convolve(image,filter,model_config,convolve_func):
    def filter_source():
        if hasattr(filter,'read'):
            filter.seek(0)
            return cPickle.loads(filter.read())
        elif is_string_like(filter):
            return cPickle.loads(open(filter).read())
        else:
            return filter
    output = {}     
    if model_config['filter']['model_name'] != 'pixels':
        for cidx in image.keys():
            output[cidx] = convolve_func(image[cidx][:,:,0],filter_source,model_config)
    else:
        for cidx in image.keys():
            output[cidx] = image[cidx][:,:,0]
            
    return output



def get_config(config_fname):
    config_path = os.path.abspath(config_fname)
    print("Config file:", config_path)
    config = {}
    execfile(config_path, {},config)
    
    return config['config']


def random_id():
    return hashlib.sha1(str(np.random.randint(10,size=(32,)))).hexdigest()    

