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
        convolve_func = numpy_convolve_images
    else:
        convolve_func = pyfft_convolve_images

    config = get_config(config_path)
    
    image_params = OrderedDict([('image',config['image'])])
    
    preproc_params = OrderedDict([('preproc' , config['preproc']),
                      ('normin' , config['normin']),
                      ('filter_kshape' , config['filter']['kshape']),
                      ('normout' , config['normout']),
                      ('pool',config['pool'])])
    preproc_params.update(config['global'])
    
    filter_params = OrderedDict([('filter',config['filter'])])
    if filter_params['filter']['model_name'] in ['random','random_gabor']:
        filter_params['filter']['id'] = random_id()
    
    activ = OrderedDict([('activ',config['activ'])]) 
    featsel = OrderedDict([('featsel',config['featsel'])])


    return [('render_image', render_image,{'args':(image_params,)}),
            ('preprocessing',preprocessing,{'params':preproc_params}),
            ('normin',normin),
            ('get_filterbank',get_filterbank,{'args':(filter_params,)}),
            ('convolve_images', convolve_func),
            ('activate',activate,{'params':activ}),
            ('normout',normout),
            ('pool',pool),
            ('postprocessing',postprocessing,{'params':featsel})]

 


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
    
@inject('v1','rendered_images',image_config_gen)
def render_image(config): 
     config = config['image'].copy()
     generator = config.pop('generator')
     if generator == 'renderman':
         return renderman_render(config)
     elif generator == 'cairo':
         return cairo_render(config)
     else:
         raise ValueError, 'image generator not recognized'
   
######image preprocessing  
@dot('v1','rendered_images',['preprocessed_images','partially_preprocessed_images'])
def preprocessing(fhs,config):
    fh = fhs[0]
    array = image2array(config,fh)
    output,orig_imga = preprocess(array,config)
    return (cPickle.dumps(output),cPickle.dumps(orig_imga))


######input norm
def norm(input,conv_mode,params):
    output = {}
    for cidx in input.keys():
       if len(input[cidx].shape) == 3:
          inobj = input[cidx]
       else:
          inobj = input[cidx][:,:,sp.newaxis]
       output[cidx] = v1f.v1like_norm(inobj, conv_mode, **params)
    return cPickle.dumps(output)
    
@dot('v1','preprocessed_images','normin_images')
def normin(fhs,config):
    fh = fhs[0]
    conv_mode = config['conv_mode']
    input = cPickle.loads(fh.read())
    return norm(input,conv_mode,config['normin'])

   
######filter generation   
def filter_config_generator(query): return [query]
    
@inject('v1','filters', filter_config_generator)
def get_filterbank(config):
    filterbank = filter_generation.get_filterbank(config)    
    return cPickle.dumps(filterbank)
    
    

######convolution
def convolve(fhs,config,convolve_func):
    image_fh = fhs[0] 
    filter_fh = fhs[1]
    image = cPickle.loads(image_fh.read())
    
    def filter_source():
        filter_fh.seek(0)
        return cPickle.loads(filter_fh.read())

    image_config = config[0] 
    filter_config = config[1] 
    output = {}
    for cidx in image.keys():
        output[cidx] = convolve_func(image[cidx][:,:,0],filter_source,image_config,filter_config)
    return cPickle.dumps(output)   

@cross('v1',['normin_images','filters'],'filtered_images')
def numpy_convolve_images(fhs,config):
    return convolve(fhs, config, v1f.v1like_filter_numpy)
    
if GPU_SUPPORT:    
    @cross('v1',['normin_images','filters'],'filtered_images',setup = v1_pyfft.setup_pyfft,cleanup = v1_pyfft.cleanup_pyfft)
    def pyfft_convolve_images(fhs,config):
        return convolve(fhs, config, v1f.v1like_filter_pyfft)


######nonlinear clipping
@dot('v1','filtered_images','activated_images')
def activate(fhs,config):
    fh = fhs[0]
    minout = config['activ']['minout'] # sustain activity
    maxout = config['activ']['maxout'] # saturation
    input = cPickle.loads(fh.read())
    output = {}
    for cidx in input.keys():
       output[cidx] = input[cidx].clip(minout, maxout)
    return cPickle.dumps(output)


######output normalization
@dot('v1','activated_images','normout_images')
def normout(fhs,config):
    fh = fhs[0]
    conv_mode = config['conv_mode']
    input = cPickle.loads(fh.read())
    return norm(input,conv_mode,config['normout'])

  
######pooling  
@dot('v1','normout_images','pooled_images')
def pool(fhs,config):
    fh = fhs[0]
    conv_mode = config['conv_mode']
    input = cPickle.loads(fh.read()) 
    output = {}
    for cidx in input.keys():
        output[cidx] = v1f.v1like_pool(input[cidx],conv_mode,**config['pool'])
    return cPickle.dumps(output)


######postprocessing    
feature_inputs = ['normin_images',
                  'filtered_images',
                  'activated_images',
                  'normout_images',
                  'pooled_images',
                  'partially_preprocessed_images']    
@dot('v1',feature_inputs,'v1_features')
def postprocessing(fhs,config):
    featsel = config['featsel']
    if any([v for (k,v) in featsel if k != 'output']):
        Normin = cPickle.loads(fhs[0].read())
        Filtered = cPickle.loads(fhs[1].read())
        Activated = cPickle.loads(fhs[2].read())
        Normout = cPickle.loads(fhs[3].read())
        Pooled = cPickle.loads(fh[4].read())
        Partially_processed = cPickle.loads(fh[5].read())
        fvector_l = postprocess(Normin,Filtered,Activated,Normout,Pooled,Partially_processed,featsel)     
    else:
        Pooled = cPickle.loads(fh[4].read())
        fvector_l = [fvector[key].ravel() for key in Pooled]
    out = sp.concatenate(fvector_l).ravel()
    
    return cPickle.dumps(out)




#=-=-=-=-=-=-=-=-=-=
#=-=-=-=-=-=-=-=-=-=
#random utilities
#=-=-=-=-=-=-=-=-=-=
#=-=-=-=-=-=-=-=-=-=

def random_id():
    hashlib.sha1(str(np.random.randint(10,size=(32,)))).hexdigest()

def get_config(config_fname):
    config_path = os.path.abspath(config_fname)
    print("Config file:", config_path)
    config = {}
    execfile(config_path, {},config)
    
    return config['config']


    



