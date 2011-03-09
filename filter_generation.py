import v1like_math as v1m
import numpy as np
import itertools

def get_filterbank(config):
    config = config['filter']
    model_name = config['model_name']
    fh, fw = config['kshape']
    
    if model_name == 'totally_random':
        num_filters = config['num_filters']
        filterbank = np.random.random((fh,fw,num_filters))
        
    elif model_name == 'random_gabor':
        num_filters = config['num_filters']
        min_wl = config['min_wavelength']
        max_wl = config['max_wavelength']  
        xc = fw/2
        yc = fh/2
        num = query['num_filters']
        filterbank = np.empty((fh,fw,num_filters))
        for i in num:
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

    return filterbank