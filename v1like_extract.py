import sys
import os
import time
from os import path
import warnings
import pprint
import hashlib
import cPickle
import warnings
import math

import numpy as np
import scipy as sp
from scipy import io

from starflow.protocols import protocolize, actualize

import v1like_funcs as v1f 
import v1like_math as v1m 
import colorconv as colorconv

from npclockit import clockit_onprofile

from dbutils import dot,cross,inject

try:
    import v1_pyfft
except:
    GPU_SUPPORT = False
else:
    GPU_SUPPORT = True
    
	@cross('v1',['normin_images','filters'],'filtered_images',setup = v1_pyfft.setup_pyfft,cleanup = v1_pyfft.cleanup_pyfft)
	def pyfft_convolve_images(fhs,config)
		image_fh = fhs[0] 
		filter_fh = fhs[1]
		image_source = lambda : cPickle.loads(image_fh.read())
		filter_source = lambda : cPickle.loads(filter_fh.read())
		image_config = config[0] 
		filter_config = config[1] 
		  
		res = v1like_filter_pyfft(img_source,filter_source,image_config,filter_config)
	  
		return cPickle.dumps(res)


@protocolize()
def v1_protocol(config_path,use_cpu = False):

    config = get_config(config_path)
    
    if use_cpu or not GPU_SUPPORT:    
        filter_convolve_func = pyfft_convolve_images
    else:
        filter_convolve_func = numpy_convolve_images
    
    image_params = config['image_query']
    preproc_params = {'preproc':config['preproc'],
	                  'normin':config['normin'],
	                  'filter':config['filter'],
	                  'normout':config['normout']}                                                             
    postproc_params = image_params.copy(); postproc_params.update(preproc_params)
    filter_params = config['filter_query']
    postconv_params = postproc_params.copy(); postconv_params.update(filter_params)
    featsel = {'featsel':config['featsel']}
    
	args = [('render_image', render_image , image_params),
	        ('im_to_array',image_to_array , image_params),
	        ('preprocess',preprocess , image_params, preproc_params),
	        ('normin',normin, postproc_params),
		    ('get_filters',get_filters, filter_params),
	        ('convolve_images', convolve_func, [postproc_params, filter_params]),
	        ('activate',activate, postconv_params),
	        ('normout',normout, postconv_params),
	        ('pool',pool, postconv_params),
	        ('add_features',add_features, next_config, featsel)]

    D = []
    for a in args:
        if len(a) == 3:
            D.append((a[0],db_update,(a[1],a[2])))
        elif len(a) == 4:
            D.append((a[0],db_update,[(a[1],a[2]),{'params':a[3]}]))
            
    actualize(D)




#=-=-=-=-=-=-=-=-=-=-=-=-=
def image_config(query):

    #get translation_x, translation_y, rotation_xy, rotation_xz, rotation_yz, lighting, color, whatever params
    
    ranger = lambda mn,mx,d : np.arange(query[mn],query[mx],query[d])
    
    tx = ranger('txmin','txmax','txdelta')
    ty = ranger('tymin','tymax','tydelta')
    tz = ranger('tzmin','tzmax','tzdelta')
    rxy = ranger('rxymin','rxymax','rxydelta')
    rxz = ranger('rxzmin','rxzmax','rxzdelta')
    ryz = ranger('ryzmin','ryzmax','ryzdelta')
    model_ids = query['model_ids']
    
    param_names = ['tx','ty','tz','rxy','rxz','ryz','model_id']
    ranges = [tz,ty,tz,rxy,xz,ryz,model_ids]
    params = [dict(zip(param_names,p)) for p in itertools.product(*ranges)]

    
    return parms
    
IMAGE_URL = 'localhost:8000/render?'


@inject('v1','rendered_images',image_config)
def render_image(config):
     
     params_list = [[config]]
     
     tmp = tempfile.mkdtemp()
     
     os.chdir(tmp)
     
     os.system('wget ' + IMAGE_URL + json.dumps(params_list))
     
     zipfile = [x for x in os.listdir('.') if x.endswith('.zip')][0]
     
     zipname = zip[:-4]
     
     os.system('tar -xzvf ' + zipfile)
     
     imagefile = [os.path.join(zipname,x) for x in os.listdir(zipname) if x.endswith('.tif')][0]
     
     return open(imagefile).read()
     
   
   
@dot('v1','rendered_images','image_arrays')
def image_to_array(fh,config):
        
    return cPickle.dumps(image2array(config,fh))
    
     
@dot('v1','image_arrays',['preprocessed_images','partially_preprocessed_images'])
def preprocess(fh,config):
    
    arr = cPickle.loads(fh.read())
    
    orig_imga,orig_imga_conv = image_preprocessing(arr,config) 
    output = {}
    for cidx in xrange(orig_imga_conv.shape[2]):
        output[cidx] = map_preprocessing(orig_imga_conv[:,:,cidx],config)
    
    return (cPickle.dumps(output),cPickle.dumps(orig_imga))

@dot('v1','preprocessed_images','normin_images')
def normin(fh,config):

    conv_mode = config['conv_mode']
    input = cPickle.loads(fh.read())
    
    return norm(input,conv_mode,config['normin'])

    
def random_id():
    hashlib.sha1(str(np.random.randint(10,size=(32,)))).hexdigest()
   

def filter_config_generator(query):
    model_name = query['model_name']

    if model_name == 'totally_random':        
        num = query['num_filters']
        fh, fw = query['kshape']
        params = [{'id':random_id(),'kshape':[fh,fw]} for i in xrange(num)]
        
        
    elif model_name == 'random_gabor':    
        
        min_wl = query['min_wavelength']
        max_wl = query['max_wavelength']
        
        
        num = query['num_filters']
        values = []
        for i in num:
            orient = 2*np.pi*np.random.random()
            freq = 1./np.random.randint(min_wl,high = max_wl)
            phase = 2*np.pi*np.random.random()
            values.append((orient,freq,phase))
       
        param_names = ['orientation','frequency','phase'] 
        params = [dict(zip(param_names,v)) for v in values]       
        
        for p in params:
            p['kshape'] = query['kshape']
       
    elif model_name == 'gridded_gabor':
        norients = query['norients']
        orients = [ o*sp.pi/norients for o in xrange(norients) ]
        divfreqs = query['divfreqs']
        freqs = [1./d for d in divfreqs]
        phases = query['phases']
        
        values = itertools.product(orients,divfreqs,phases)
        
        param_names = ['orientation','frequency','phase']
        params = [dict(zip(param_names,v)) for v in values]
        for p in params:
            p['kshape'] = query['kshape']
                 
       
    configs = [{'model_name':model_name,'params':p} for p in params]
    
    return configs
 
 
    
@inject('v1','filters', filter_config_generator)
def get_filters(config):

    model_name = config['model_name']
    fh, fw = config['kshape']
    
    if model_name == 'totally_random':
        filt = np.random.random((fh,fw))
        
    elif model_name = 'random_gabor' or 'gridded_gabor':
        freq = config['frequency']
        orient = config['orientation']
        phase = config['phase']
        xc = fw/2
        yc = fh/2
        filt = v1m.gabor2d(xc,yc,xc,yc,
                               freq,orient,phase,
                               (fw,fh))
        
        
    
    return cPickle.dumps(filt)
    

@cross('v1',['normin_images','filters'],'filtered_images')
def numpy_convolve_images(fhs,config)

    image_fh = fhs[0] 
    filter_fh = fhs[1]
    image_source = lambda x : cPickle.loads(image_fh.read())
    filter_source = lambda x : cPickle.loads(filter_fh.read())
    image_config = config[0] 
    filter_config = config[1] 
      
    res = v1like_filter_numpy(img_source,filter_source,image_config,filter_config)
  
    return cPickle.dumps(res)
    


    
    
@dot('v1','filtered_images','activated_images')
def activate(fh,config):
    minout = config['activ']['minout'] # sustain activity
    maxout = config['activ']['maxout'] # saturation
    input = cPickle.loads(fh.read())
    
    output = {}
    for cidx in input.keys():
       output[cidx] = input[cidx].clip(minout, maxout)
       
    return cPickle.dumps(output)


@dot('v1','activated_images','normout_images')
def normout(fh,config):
    conv_mode = config['conv_mode']
    input = cPickle.loads(fh.read())
    return norm(input,conv_mode,config['normout'])

        
@dot('v1','normout_images','pooled_images')
def pool(fh,config):
    conv_mode = config['conv_mode']
    input = cPickle.loads(fh.read())
    
    output = {}
    for cidx in input.keys():
        output[cidx] = v1f.v1like_pool(input[cidx],conv_mode,**config['pool'])
        
    return cPickle.dumps(output)

    
feature_inputs = ['normin_images',
                  'filtered_images',
                  'activated_images',
                  'pooled_images',
                  'partially_preprocessed_images']    
@dot('v1',feature_inputs,'final_processed_features')
def add_features(fhs,config):
 
    featsel = config['featsel']
    
    if featsel['output']:
        imga1 = cPickle.loads(fhs[0].read())
        imga2 = cPickle.loads(fhs[1].read())
        imga3 = cPickle.loads(fhs[2].read())
        imga5 = cPickle.loads(fh[3].read())
        orig_imga = cPickle.load(fh[4])
            
        keys = imga5.keys()
        fvector_l = []
        for cidx in keys:
            fvector = include_map_level_features(imga1[cidx],
                                                 imga2[cidx],
                                                 imga3[cidx],
                                                 imga4[cidx],
                                                 imga5[cidx],
                                                 imga5[cidx],
                                                 featsel)
            fvector_l += [fvector]
    
        fvector_l = include_image_level_features(orig_imga,fvector_l,featsel)
        
        fvector_l = [fvector.ravel() for fvector in fvector_l]
        
    else:
        imga5 = cPickle.loads(fh[3].read())
        fvector_l = [fvector[key].ravel() for key in imga5]
    
    out = sp.concatenate(fvector_l).ravel()
    
    return cPickle.dumps(out)


#=-=-=-=-=-=-=-=-=-=



def get_config(config_fname):
    config_path = path.abspath(config_fname)
    if verbose: print "Config file:", config_path
    config = {}
    execfile(config_path, {},config)
    
    return config

    
#=-=-=-=-=-=-=-=-=-=-

def norm(input,conv_mode,params):
    output = {}
    for cidx in input.keys():
       if len(input[cidx].shape) == 3:
          inobj = input[cidx]
       else:
          inobj = input[cidx][:,:,sp.newaxis]
       output[cidx] = v1f.v1like_norm(inobj, conv_mode, **params)

    return cPickle.dumps(output)


def image2array(rep,fobj):
    resize_type = rep['preproc'].get('resize_type', 'input')
    if resize_type == 'output':
        if 'max_edge' not in rep['preproc']:
            raise NotImplementedError
        # add whatever is needed to get output = max_edge
        new_max_edge = rep['preproc']['max_edge']

        preproc_lsum = rep['preproc']['lsum_ksize']
        new_max_edge += preproc_lsum-1
            
        normin_kshape = rep['normin']['kshape']
        assert normin_kshape[0] == normin_kshape[1]
        new_max_edge += normin_kshape[0]-1

        filter_kshape = rep['filter']['kshape']
        assert filter_kshape[0] == filter_kshape[1]
        new_max_edge += filter_kshape[0]-1
        
        normout_kshape = rep['normout']['kshape']
        assert normout_kshape[0] == normout_kshape[1]
        new_max_edge += normout_kshape[0]-1
        
        pool_lsum = rep['pool']['lsum_ksize']
        new_max_edge += pool_lsum-1

        rep['preproc']['max_edge'] = new_max_edge
    
    if 'max_edge' in rep['preproc']:
        max_edge = rep['preproc']['max_edge']
        resize_method = rep['preproc']['resize_method']
        imgarr = v1f.get_image(fobj, max_edge=max_edge,
                           resize_method=resize_method)
    else:
        resize = rep['preproc']['resize']
        resize_method = rep['preproc']['resize_method']        
        imgarr = v1f.get_image2(fobj, resize=resize,
                            resize_method=resize_method)
                            
    return imgarr
    
def image_preprocessing(arr,params):

    arr = sp.atleast_3d(arr)

    smallest_edge = min(arr.shape[:2])

    rep = params
    
    preproc_lsum = rep['preproc']['lsum_ksize']
    if preproc_lsum is None:
        preproc_lsum = 1
    smallest_edge -= (preproc_lsum-1)
            
    normin_kshape = rep['normin']['kshape']
    smallest_edge -= (normin_kshape[0]-1)

    filter_kshape = rep['filter']['kshape']
    smallest_edge -= (filter_kshape[0]-1)
        
    normout_kshape = rep['normout']['kshape']
    smallest_edge -= (normout_kshape[0]-1)
        
    pool_lsum = rep['pool']['lsum_ksize']
    smallest_edge -= (pool_lsum-1)

    arrh, arrw, _ = arr.shape

    if smallest_edge <= 0 and rep['conv_mode'] == 'valid':
        if arrh > arrw:
            new_w = arrw - smallest_edge + 1
            new_h =  int(np.round(1.*new_w  * arrh/arrw))
            print new_w, new_h
            raise
        elif arrh < arrw:
            new_h = arrh - smallest_edge + 1
            new_w =  int(np.round(1.*new_h  * arrw/arrh))
            print new_w, new_h
            raise
        else:
            pass
    
    # TODO: finish image size adjustment
    assert min(arr.shape[:2]) > 0

    # use the first 3 channels only
    orig_imga = arr.astype("float32")[:,:,:3]

    # make sure that we don't have a 3-channel (pseudo) gray image
    if orig_imga.shape[2] == 3 \
            and (orig_imga[:,:,0]-orig_imga.mean(2) < 0.1*orig_imga.max()).all() \
            and (orig_imga[:,:,1]-orig_imga.mean(2) < 0.1*orig_imga.max()).all() \
            and (orig_imga[:,:,2]-orig_imga.mean(2) < 0.1*orig_imga.max()).all():
        orig_imga = sp.atleast_3d(orig_imga[:,:,0])

    # rescale to [0,1]
    #print orig_imga.min(), orig_imga.max()
    if orig_imga.min() == orig_imga.max():
        raise MinMaxError("[ERROR] orig_imga.min() == orig_imga.max() "
                          "orig_imga.min() = %f, orig_imga.max() = %f"
                          % (orig_imga.min(), orig_imga.max())
                          )
    
    orig_imga -= orig_imga.min()
    orig_imga /= orig_imga.max()

    # -- color conversion
    # insure 3 dims
    #print orig_imga.shape
    if orig_imga.ndim == 2 or orig_imga.shape[2] == 1:
        orig_imga_new = sp.empty(orig_imga.shape[:2] + (3,), dtype="float32")
        orig_imga.shape = orig_imga_new[:,:,0].shape
        orig_imga_new[:,:,0] = 0.2989*orig_imga
        orig_imga_new[:,:,1] = 0.5870*orig_imga
        orig_imga_new[:,:,2] = 0.1141*orig_imga
        orig_imga = orig_imga_new    


    if params['color_space'] == 'rgb':
        orig_imga_conv = orig_imga
#     elif params['color_space'] == 'rg':
#         orig_imga_conv = colorconv.rg_convert(orig_imga)
    elif params['color_space'] == 'rg2':
        orig_imga_conv = colorconv.rg2_convert(orig_imga)
    elif params['color_space'] == 'gray':
        orig_imga_conv = colorconv.gray_convert(orig_imga)
        orig_imga_conv.shape = orig_imga_conv.shape + (1,)
    elif params['color_space'] == 'opp':
        orig_imga_conv = colorconv.opp_convert(orig_imga)
    elif params['color_space'] == 'oppnorm':
        orig_imga_conv = colorconv.oppnorm_convert(orig_imga)
    elif params['color_space'] == 'chrom':
        orig_imga_conv = colorconv.chrom_convert(orig_imga)
#     elif params['color_space'] == 'opponent':
#         orig_imga_conv = colorconv.opponent_convert(orig_imga)
#     elif params['color_space'] == 'W':
#         orig_imga_conv = colorconv.W_convert(orig_imga)
    elif params['color_space'] == 'hsv':
        orig_imga_conv = colorconv.hsv_convert(orig_imga)
    else:
        raise ValueError, "params['color_space'] not understood"
        
    return orig_imga,orig_imga_conv
    

def map_preprocessing(imga0,params): 
    
    assert(imga0.min() != imga0.max())
    
    # -- 0. preprocessing
    #imga0 = imga0 / 255.0
    
    # flip image ?
    if 'flip_lr' in params['preproc'] and params['preproc']['flip_lr']:
        imga0 = imga0[:,::-1]
        
    if 'flip_ud' in params['preproc'] and params['preproc']['flip_ud']:
        imga0 = imga0[::-1,:]            
    
    # smoothing
    lsum_ksize = params['preproc']['lsum_ksize']
    conv_mode = params['conv_mode']
    if lsum_ksize is not None:
         k = sp.ones((lsum_ksize), 'f') / lsum_ksize             
         imga0 = conv(conv(imga0, k[sp.newaxis,:], conv_mode), 
                      k[:,sp.newaxis], conv_mode)
         
    # whiten full image (assume True)
    if 'whiten' not in params['preproc'] or params['preproc']['whiten']:
        imga0 -= imga0.mean()
        if imga0.std() != 0:
            imga0 /= imga0.std()

    return imga0

     
def include_image_level_features(orig_imga,fvector_l,featsel):
    # include grayscale values ?
    f_input_gray = featsel['input_gray']
    if f_input_gray is not None:
        shape = f_input_gray
        #print orig_imga.shape
        fvector_l += [sp.misc.imresize(colorconv.gray_convert(orig_imga), shape).ravel()]

    # include color histograms ?
    f_input_colorhists = featsel['input_colorhists']
    if f_input_colorhists is not None:
        nbins = f_input_colorhists
        colorhists = sp.empty((3,nbins), 'f')
        if orig_imga.ndim == 3:
            for d in xrange(3):
                h = sp.histogram(orig_imga[:,:,d].ravel(),
                                 bins=nbins,
                                 range=[0,255])
                binvals = h[0].astype('f')
                colorhists[d] = binvals
        else:
            raise ValueError, "orig_imga.ndim == 3"
            #h = sp.histogram(orig_imga[:,:].ravel(),
            #                 bins=nbins,
            #                 range=[0,255])
            #binvals = h[0].astype('f')
            #colorhists[:] = binvals

        #feat_l += [colorhists.ravel()]
        fvector_l += [colorhists.ravel()]

    return fvector_l
    

def include_map_level_features(imga1,imga2,imga3,imga4,imga5,output,featsel):
    feat_l = []

    # include input norm histograms ? 
    f_normin_hists = featsel['normin_hists']
    if f_normin_hists is not None:
        division, nfeatures = f_norminhists
        feat_l += [v1f.rephists(imga1, division, nfeatures)]

    # include filter output histograms ? 
    f_filter_hists = featsel['filter_hists']
    if f_filter_hists is not None:
        division, nfeatures = f_filter_hists
        feat_l += [v1f.rephists(imga2, division, nfeatures)]

    # include activation output histograms ?     
    f_activ_hists = featsel['activ_hists']
    if f_activ_hists is not None:
        division, nfeatures = f_activ_hists
        feat_l += [v1f.rephists(imga3, division, nfeatures)]

    # include output norm histograms ?     
    f_normout_hists = featsel['normout_hists']
    if f_normout_hists is not None:
        division, nfeatures = f_normout_hists
        feat_l += [v1f.rephists(imga4, division, nfeatures)]

    # include representation output histograms ? 
    f_pool_hists = featsel['pool_hists']
    if f_pool_hists is not None:
        division, nfeatures = f_pool_hists
        feat_l += [v1f.rephists(imga5, division, nfeatures)]

    # include representation output ?
    f_output = featsel['output']
    if f_output and len(feat_l) != 0:
        fvector = sp.concatenate([output.ravel()]+feat_l)
    else:
        fvector = output    
   
    return fvector   

