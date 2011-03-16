import cPickle

import numpy as np

@aggregate('v1','raw_images','raw_images','models')
def learn_filterbank(fhs,configs):
    fhs = [fh[0] for fh in fhs]
    
    config = configs[0]['model'].copy()
    num_slices = config['num_slices']  
    filterbank = get_initial_filterbank(config)
    
    fh,fw,num_filters = filterbank.shape
    filter_kshape = (fh,fw)
    
    config['round'] = 0
    counts = np.zeros(num_filters)
    
    for fh in fhs:
        array = image2array(model_config,image_fh)
        
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
            lr = 1./counts[imax]
            filterbank[:,:,i] = filterbank[:,:,i]*(1 - lr) + patch*lr
            
            
            
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
    