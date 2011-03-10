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
        postfilter_func = numpy_postfilter
    else:
        postfilter_func = pyfft_postfilter

    config = get_config(config_path)
    
    image_params = OrderedDict([('image',config['image'])])
    
    prefilter_params = OrderedDict([('preproc' , config['preproc']),
                      ('normin' , config['normin']),
                      ('filter_kshape' , config['filter']['kshape']),
                      ('normout' , config['normout']),
                      ('pool',config['pool'])])
    prefilter_params.update(config['global'])
    
    filter_params = OrderedDict([('filter',config['filter'])])
    if filter_params['filter']['model_name'] in ['random','random_gabor']:
        filter_params['filter']['id'] = random_id()
    
    postfilter_params = OrderedDict([('activ',config['activ']),('featsel',config['featsel'])]) 

    return [('render_image', render_image,{'args':(image_params,)}), 
            ('prefiltering',prefilter,{'params':prefilter_params}),                        
            ('get_filterbank',get_filterbank,{'args':(filter_params,)}),            
            ('postfiltering',postfilter_func,{'params':postfilter_params})]



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
   
   
   
######prefiltering 
def norm(input,conv_mode,params):
    output = {}
    for cidx in input.keys():
       if len(input[cidx].shape) == 3:
          inobj = input[cidx]
       else:
          inobj = input[cidx][:,:,sp.newaxis]
       output[cidx] = v1f.v1like_norm(inobj, conv_mode, **params)
    return output


@dot('v1','rendered_images','prefiltered_images')
def prefilter(fhs,config):
    fh = fhs[0]
    array = image2array(config,fh)
    output,orig_imga = preprocess(array,config)
    conv_mode = config['conv_mode']
    normed_output = norm(output,conv_mode,config['normin'])
    return cPickle.dumps([orig_imga,normed_output])

   
######filter generation   
def filter_config_generator(query): return [query]
    
@inject('v1','filters', filter_config_generator)
def get_filterbank(config):
    filterbank = filter_generation.get_filterbank(config)    
    return cPickle.dumps(filterbank)
    
    

######postfiltering

def convolve(image,filter_fh,image_config,filter_config,convolve_func):

    
    def filter_source():
        filter_fh.seek(0)
        return cPickle.loads(filter_fh.read())


    output = {}
    for cidx in image.keys():
        output[cidx] = convolve_func(image[cidx][:,:,0],filter_source,image_config,filter_config)
    return output
    

def postfilter(fhs,config,convolve_func):
    image_fh = fhs[0] 
    filter_fh = fhs[1]
    orig_imga, norm_in = cPickle.loads(image_fh.read())

    image_config = config[0] 
    filter_config = config[1] 
    params = config[2]

    conv_mode = image_config['conv_mode'] 

    filtered = convolve(norm_in, filter_fh, image_config, filter_config, convolve_func)

    minout = params['activ']['minout'] # sustain activity
    maxout = params['activ']['maxout'] # saturation
    activ = {}
    for cidx in filtered.keys():
       activ[cidx] = filtered[cidx].clip(minout, maxout)
       
    norm_out = norm(activ,conv_mode,image_config['normout'])
    
    pooled = {}
    for cidx in norm_out.keys():
        pooled[cidx] = v1f.v1like_pool(norm_out[cidx],conv_mode,**image_config['pool'])
        
    featsel = params['featsel']    

    fvector_l = postprocess(norm_in,filtered,activ,norm_out,pooled,orig_imga,featsel) 

    out = sp.concatenate(fvector_l).ravel()
    
    return cPickle.dumps(out)


@cross('v1',['prefiltered_images','filters'],'v1_features')
def numpy_postfilter(fhs,config):
    return postfilter(fhs,config,v1f.v1like_filter_numpy)
    
if GPU_SUPPORT:    
    @cross('v1',['prefiltered_images','filters'],'v1_features',setup = v1_pyfft.setup_pyfft,cleanup = v1_pyfft.cleanup_pyfft)
    def pyfft_postfilter(fhs,config):
        return postfilter(fhs, config, v1f.v1like_filter_pyfft)





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


    



