import v1like_math as v1m
import numpy as np
import itertools
import rendering
import processing

def normalize(Y):
    m = Y.mean()
    return (Y - m) / np.sqrt(((Y - m)**2).sum())

def get_filterbank(config):
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
        min_wl = config['min_wavelength']
        max_wl = config['max_wavelength']  
        xc = fw/2
        yc = fh/2
        filterbank = np.empty((fh,fw,num_filters))
        for i in range(num_filters):
            orient = 2*np.pi*np.random.random()
            freq = 1./np.random.randint(min_wl,high = max_wl)
            phase = 2*np.pi*np.random.random()
            filterbank[:,:,i] = v1m.gabor2d(xc,yc,xc,yc,
                               freq,orient,phase,
                               (fw,fh))     
                               
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
    
    elif model_name == 'cairo_generated':
        specs = config.get('specs')
        if not specs:
            specs = [spec['image'] for spec in rendering.cairo_config_gen(config['spec_gen'])]
        filterbank = np.empty((fh,fw,len(specs)))
        for (i,spec) in enumerate(specs):
            im_fh = rendering.cairo_render(spec,returnfh=True)
            arr = processing.image2array({},im_fh)
            arr = arr[:,:,:2].max(2)
            arrx0 = arr.shape[0]/2
            arry0 = arr.shape[1]/2
            dh = fh/2; dw = fw/2
            filterbank[:,:,i] = normalize(arr[arrx0-dh:arrx0+(fh - dh),arry0-dw:arry0+(fw-dw)])
                   

    return filterbank