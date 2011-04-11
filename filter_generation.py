import v1like_math as v1m
import v1like_funcs as v1f
import numpy as np
import itertools
import rendering
import processing
from bson import SON

def normalize(Y):
    m = Y.mean()
    return (Y - m) / np.sqrt(((Y - m)**2).sum())
    
import rendering, processing
import scipy as sp

def center_surround_orth(model_config):

    conv_mode = model_config['conv_mode']
    
    L = np.empty(tuple(model_config['filter']['kshape']) + (2*len( model_config['filter']['base_images'] ),))
    
    for (ind,image_config) in enumerate(model_config['filter']['base_images']):
        image_fh = rendering.cairo_render(image_config,returnfh=True)
        
        #preprocessing
        array = processing.image2array(model_config ,image_fh)
      
        preprocessed,orig_imga = processing.preprocess(array,model_config )
                
        norm_in = norm(preprocessed,conv_mode,model_config.get('normin'))
    
        array = make_array(norm_in) 
        array = array[:,:,:2].max(2).astype(np.float)
        
        arr_box = cut_bounding_box(array)
        s = model_config['filter']['kshape'] 
        
        X = np.zeros(s)

        (hx,wx) = X.shape
        (ha,wa) = arr_box.shape


        hx0 = max((hx - ha) / 2, 0)
        hx1 = min(ha + hx0,hx)
        ha0 = max((ha - hx) / 2, 0)
        ha1 = min(hx + ha0, ha)
        wx0 = max((wx - wa) / 2, 0)
        wx1 = min(wa + wx0,wx)
        wa0 = max((wa - wx) / 2, 0)
        wa1 = min(wx + wa0, wa)       
        
        X[hx0:hx1, wx0:wx1] = arr_box[ha0:ha1, wa0:wa1]
          
        X = normalize(X)
        
        L[:,:,2*ind] = X
        L[:,:,2*ind + 1] = X.T
        
    return L

def center_surround(model_config):
    conv_mode = model_config['conv_mode']
    
    L = np.empty(tuple(model_config['filter']['kshape']) + (len( model_config['filter']['base_images'] ),))
    
    for (ind,image_config) in enumerate(model_config['filter']['base_images']):
        image_fh = rendering.cairo_render(image_config,returnfh=True)
        
        #preprocessing
        array = processing.image2array(model_config ,image_fh)
      
        preprocessed,orig_imga = processing.preprocess(array,model_config )
                
        norm_in = norm(preprocessed,conv_mode,model_config.get('normin'))
    
        array = make_array(norm_in) 
        array = array[:,:,:2].max(2).astype(np.float)
        
        arr_box = cut_bounding_box(array)
        s = model_config['filter']['kshape'] 
        
        X = np.zeros(s)
        
        (hx,wx) = X.shape
        (ha,wa) = arr_box.shape

        hx0 = max((hx - ha) / 2, 0)
        hx1 = min(ha + hx0,hx)
        ha0 = max((ha - hx) / 2, 0)
        ha1 = min(hx + ha0, ha)
        wx0 = max((wx - wa) / 2, 0)
        wx1 = min(wa + wx0,wx)
        wa0 = max((wa - wx) / 2, 0)
        wa1 = min(wx + wa0, wa)       
        
        X[hx0:hx1, wx0:wx1] = arr_box[ha0:ha1, wa0:wa1]
  
        X = normalize(X)        
        L[:,:,ind] = X

        
    return L
       
   
    
def make_array(x):
    s = x[0].shape[:2]
    K = x.keys()  
    K.sort()
    K = K[:1]
    arr = np.empty(s + (len(K),))
    
    for k in K:
        arr[:,:,k] = x[k][:,:,0]
    return arr

def cut_bounding_box(arr,theta = None,buffer = 0):
    if theta is not None:
        arr = np.where(arr > theta,arr,0)
        
    xnz = arr.sum(1).nonzero()[0]
    if len(xnz) > 0:
        x0,x1 = xnz[0],xnz[-1]
        ynz = arr.sum(0).nonzero()[0]
        y0,y1 = ynz[0],ynz[-1]
    
        return arr[x0-buffer:x1+1+buffer,y0-buffer:y1+1+buffer]
    else:
        raise ValueError, 'Empty Bounding box'


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


def get_filterbank(config):
    model_config = config
    config = config['filter']
    model_name = config['model_name']
    fh, fw = config['kshape']
    
    if model_name == 'really_random':
        num_filters = config['num_filters']
        filterbank = np.random.random((fh,fw,num_filters))
        if config.get('normalize',True):
            for i in range(filterbank.shape[2]):
                filterbank[:,:,i] = normalize(filterbank[:,:,i])
        
    elif model_name == 'random_gabor':
        num_filters = config['num_filters']
        xc = fw/2
        yc = fh/2
        filterbank = np.empty((fh,fw,num_filters))
        orients = []
        freqs = []
        phases = []
        df = config.get('divfreq')
        for i in range(num_filters):
            orient = config.get('orient',2*np.pi*np.random.random())
            orients.append(orient)
            if not df:
                freq = 1./np.random.randint(config['min_wavelength'],high = config['max_wavelength'])
            else:
                freq = 1./df
            freqs.append(freq)
            phase = config.get('phase',2*np.pi*np.random.random())
            phases.append(phase)
            
            filterbank[:,:,i] = v1m.gabor2d(xc,yc,xc,yc,
                               freq,orient,phase,
                               (fw,fh))   
        
        return SON([('filterbank',filterbank),('orients',orients),('phases',phases),('freqs',freqs)])
                               
    elif model_name == 'gridded_gabor':
        norients = config['norients']
        orients = [ o*np.pi/norients for o in xrange(norients) ]
        divfreqs = config['divfreqs']
        freqs = [1./d for d in divfreqs]
        phases = config['phases']       
        xc = fw/2
        yc = fh/2
        values = list(itertools.product(freqs,orients,phases))
        num_filters = len(values)
        filterbank = np.empty((fh,fw,num_filters))
        for (i,(freq,orient,phase)) in enumerate(values):
            filterbank[:,:,i] = v1m.gabor2d(xc,yc,xc,yc,
                               freq,orient,phase,
                               (fw,fh)) 
                               
    elif model_name == 'pixels':
        return np.ones((fh,fw,1))

    elif model_name == 'specific_gabor':
        orients = config['orients']
        divfreqs = config['divfreqs']
        phases = config['phases']
        xc = fw/2
        yc = fh/2
        freqs = [1./d for d in divfreqs]
        values = zip(freqs,orients,phases)
        num_filters = len(values)
        filterbank = np.empty((fh,fw,num_filters))
        for (i,(freq,orient,phase)) in enumerate(values):
            filterbank[:,:,i] = v1m.gabor2d(xc,yc,xc,yc,
                               freq,orient,phase,
                               (fw,fh)) 
        
    
    elif model_name == 'cairo_generated':
        specs = config.get('specs')
        if not specs:
            specs = [spec['image'] for spec in rendering.cairo_config_gen(config['spec_gen'])]
        filterbank = np.empty((fh,fw,len(specs)))
        for (i,spec) in enumerate(specs):
            im_fh = rendering.cairo_render(spec,returnfh=True)
            arr = processing.image2array({'color_space':'rgb'},im_fh).astype(np.int32)
            arr = arr[:,:,0] - arr[:,:,1]
            arrx0 = arr.shape[0]/2
            arry0 = arr.shape[1]/2
            dh = fh/2; dw = fw/2
            filterbank[:,:,i] = normalize(arr[arrx0-dh:arrx0+(fh - dh),arry0-dw:arry0+(fw-dw)])
    
    elif model_name == 'center_surround':
        
        if config.get('orth',True):
            return center_surround_orth(model_config)
        else:
            return center_surround(model_config)

    return filterbank