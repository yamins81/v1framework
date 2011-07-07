import v1like_math as v1m
import v1like_funcs as v1f
import numpy as np
import itertools
import rendering
import processing
from bson import SON
import cPickle

def normalize(Y):
    m = Y.mean()
    return (Y - m) / np.sqrt(((Y - m)**2).sum())
    
import rendering, processing
import scipy as sp

def center_surround_orth(model_config):

    conv_mode = model_config['conv_mode']
    
    L = np.empty((2*len( model_config['filter']['base_images'] ),) + tuple(model_config['filter']['kshape']) )
    
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
        
        L[2*ind,:,:] = X
        L[2*ind + 1,:,:] = X.T
        
    return L

def center_surround(model_config):
    conv_mode = model_config['conv_mode']
    
    L = np.empty((len( model_config['filter']['base_images'] ),) + tuple(model_config['filter']['kshape']) )
    
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
        L[ind,:,:] = X

        
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



def get_hierarchical_filterbanks(config):

    filterbanks = [None]
    filterbank1 = get_filterbank(config[1])
    filterbanks.append(filterbank1)
    n1 = len(filterbank1)
    if len(config) > 2:
        configL2 = config[2]
        if configL2['filter']['model_name'] == 'uniform':
            (fh,fw) = configL2['filter']['ker_shape']
            f1 = config[1]['filter']
            assert f1['model_name'] == 'gridded_gabor'
            norients = f1['norients']
            
            fsample = configL2['filter'].get('fsample',1)
            osample = configL2['filter'].get('osample',1)
            
            freq2 = len(f1['divfreqs'])/fsample
            or2 = f1['norients']/osample
            
            n2 = freq2*or2
            
            fbank = np.zeros((n2*(n2 - 1 )/2,fh,fw,n1))
            fnum = 0
            for i in range(n2):
                for j in range(i+1,n2):
                    fb1 = i/or2; orb1 = i - or2*(i/or2)
                    fb2 = j/or2; orb2 = j - or2*(j/or2)
                    
                    freqs1 = range(fb1*fsample,(fb1+1)*fsample)
                    ors1 = range(orb1*osample,(orb1+1)*osample)
                    
                    freqs2 = range(fb2*fsample,(fb2+1)*fsample)
                    ors2 = range(orb2*osample,(orb2+1)*osample)
                    
                    I = [norients*f + o for f in freqs1 for o in ors1]
                    J = [norients*f + o for f in freqs2 for o in ors2]
                    for ind in I + J:
                        fbank[fnum,:,:,ind] = 1
                    fbank[fnum] = normalize(fbank[fnum])
                    fnum += 1
            
            filterbanks.append(fbank)       
        elif configL2['filter']['model_name'] == 'freq_uniform':
            (fh,fw) = configL2['filter']['ker_shape']
            f1 = config[1]['filter']
            assert f1['model_name'] == 'gridded_gabor'
            norients = f1['norients']
            
            osample = configL2['filter'].get('osample',1)
            
            freq2 = len(f1['divfreqs'])
            or2 = f1['norients']/osample
                    
            fbank = np.zeros((freq2*or2**2,fh,fw,n1))
            fnum = 0
            for i in range(freq2):
                for j in range(or2):
                    for k in range(or2):            
                        ors1 = range(j*osample,(j+1)*osample)
                        ors2 = range(k*osample,(k+1)*osample)

                        I = [norients*i + o for o in ors1]
                        J = [norients*i + o for o in ors2]    
                        for ind in I + J:
                            fbank[fnum,:,:,ind] = 1
                        fbank[fnum] = normalize(fbank[fnum])
                        fnum += 1
            
            filterbanks.append(fbank)       
        elif configL2['filter']['model_name'] == 'correlation':
            import pymongo
            image_spec = configL2['filter']['images']
            model_spec = config[:2]
            task_spec = configL2['filter']['task']
            conn = pymongo.Connection(document_class=SON)
            db = conn['thor']
            coll = db['correlation_extraction.files']
            import gridfs
            fs = gridfs.GridFS(db,'correlation_extraction')
            fn = coll.find_one({'model.layers':model_spec,'images':image_spec,'task':task_spec})['filename']
            fh,fw = coll.find_one({'filename':fn})['task']['ker_shape']
            V,M = cPickle.loads(fs.get_version(fn).read())['sample_result']
            if configL2['filter'].get('random_subset'):
                const = configL2['filter']['random_subset']['const']
                R = np.random.binomial(1,const,V.shape)
                V = R * V 
            Z = np.zeros(M.shape)
            N = configL2['filter']['num_filters']
            s = (fh,fw,n1)
            filterbank = np.empty((N,) + s)
            for ind in range(N):
                print('Sample %d ...' % ind)
                filter = np.random.multivariate_normal(Z,V)
                #filter = np.random.multivariate_normal(M,V)
                filter = normalize(filter.reshape(s))
                filterbank[ind] = filter            
            filterbanks.append(filterbank)
        elif configL2['filter']['model_name'] == 'eigenstat':
            import pymongo
            image_spec = configL2['filter']['images']
            model_spec = config[:2]
            task_spec = configL2['filter']['task']
            conn = pymongo.Connection(document_class=SON)
            db = conn['thor']
            coll = db['correlation_extraction.files']
            #fn = configL12['filter']['corr_filename']
            import gridfs
            fs = gridfs.GridFS(db,'correlation_extraction')
            fn = coll.find_one({'model.layers':model_spec,'images':image_spec,'task':task_spec})['filename']
            fh,fw = coll.find_one({'filename':fn})['task']['ker_shape']
            V,M = cPickle.loads(fs.get_version(fn).read())['sample_result']
            print('computing eigenvectors ...')
            Vals,Vecs = np.linalg.eig(V)
            print('...done')
            N = configL2['filter']['num_filters']
            s = (fh,fw,n1)
            filterbank = np.empty((N,) + s)
            anti = configL2['filter'].get('anti',False)
            for ind in range(N):
                if anti:
                    filter = Vecs[:,-ind]            
                else:
                    filter = Vecs[:,ind]
                filter = normalize(filter.reshape(s))
                filterbank[ind] = filter            
            filterbanks.append(filterbank)
        elif configL2['filter']['model_name'] == 'really_random':
            n2 = configL2['filter']['num_filters']
            (fh,fw) = configL2['filter']['ker_shape']
            filterbank = get_random_filterbank((n2,fh,fw,n1),normalization=configL2['filter'].get('normalize',True))
            filterbanks.append(filterbank)
        elif configL2['filter']['model_name'] == 'multiply':
            if (not configL2['filter'].get('sum_up',True)) or config[1]['filter']['model_name'] == 'really_random':
                filterbanks.append((None,None))
            else:             
                filterbanks.append((len(config[1]['filter']['divfreqs']),config[1]['filter']['norients']))
        elif configL2['filter']['model_name'] == 'gridded_gabor':
            norients = configL2['filter']['norients']
            orients = [ (o1*np.pi/norients,o2*np.pi/norients) for o1 in range(norients) for o2 in range(norients) ]
            divfreqs = configL2['filter']['divfreqs'] 
            freqs = [1./d for d in divfreqs]
            phases = configL2['filter']['phases']       
            (fh,fw) = configL2['filter']['ker_shape']
            xc = fw/2
            yc = fh/2
            zc = n1/2
            values = list(itertools.product(freqs,orients,phases))
            num_filters = len(values)
            filterbank = np.empty((num_filters,fh,fw,n1))
            for (i,(freq,orient,phase)) in enumerate(values):
                filterbank[i] = v1m.gabor3d(xc,yc,zc,xc,yc,zc,
                                   freq,orient,phase,
                                   (fw,fh,n1)) 
            filterbanks.append(filterbank)
        else:
            filterbanks.append(get_filterbank(configL2))
        for c in config[3:]:
            configL3 = config[3]
            if configL3['filter']['model_name'] == 'really_random':
                n3 = configL3['filter']['num_filters']
                (fh,fw) = configL3['filter']['ker_shape']
                filterbank = get_random_filterbank((n3,fh,fw,n2),normalization=configL3['filter'].get('normalize',True))
                filterbanks.append(filterbank)
            else:
                filterbanks.append(get_filterbank(c))
    
    for (ind,fb) in enumerate(filterbanks):
        if fb is not None:
            filterbanks[ind] = np.cast[np.float32](fb)
    return filterbanks
        

def get_random_filterbank(s,normalization=True):
    filterbank = np.random.random(s)
    if normalization:
        for i in range(filterbank.shape[0]):
            filterbank[i] = normalize(filterbank[i])
    return filterbank
    

def get_filterbank(config):

    model_config = config
    config = config['filter']
    model_name = config['model_name']
    fh, fw = config.get('kshape',config.get('ker_shape'))
    
    if model_name == 'really_random':
        num_filters = config['num_filters']
        filterbank = get_random_filterbank((num_filters,fh,fw),normalization=config.get('normalize',True))

    elif model_name == 'random_gabor':
        num_filters = config['num_filters']
        xc = fw/2
        yc = fh/2
        filterbank = np.empty((num_filters,fh,fw))
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
            
            filterbank[i,:,:] = v1m.gabor2d(xc,yc,xc,yc,
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
        filterbank = np.empty((num_filters,fh,fw))
        for (i,(freq,orient,phase)) in enumerate(values):
            filterbank[i,:,:] = v1m.gabor2d(xc,yc,xc,yc,
                               freq,orient,phase,
                               (fw,fh)) 
                               
    elif model_name == 'pixels':
        return np.ones((1,fh,fw))

    elif model_name == 'specific_gabor':
        orients = config['orients']
        divfreqs = config['divfreqs']
        phases = config['phases']
        xc = fw/2
        yc = fh/2
        freqs = [1./d for d in divfreqs]
        values = zip(freqs,orients,phases)
        num_filters = len(values)
        filterbank = np.empty((num_filters,fh,fw))
        for (i,(freq,orient,phase)) in enumerate(values):
            filterbank[i,:,:] = v1m.gabor2d(xc,yc,xc,yc,
                               freq,orient,phase,
                               (fw,fh)) 
        
    
    elif model_name == 'cairo_generated':
        specs = config.get('specs')
        if not specs:
            specs = [spec['image'] for spec in rendering.cairo_config_gen(config['spec_gen'])]
        filterbank = np.empty((len(specs),fh,fw))
        for (i,spec) in enumerate(specs):
            im_fh = rendering.cairo_render(spec,returnfh=True)
            arr = processing.image2array({'color_space':'rgb'},im_fh).astype(np.int32)
            arr = arr[:,:,0] - arr[:,:,1]
            arrx0 = arr.shape[0]/2
            arry0 = arr.shape[1]/2
            dh = fh/2; dw = fw/2
            filterbank[i,:,:] = normalize(arr[arrx0-dh:arrx0+(fh - dh),arry0-dw:arry0+(fw-dw)])
    
    elif model_name == 'center_surround':
        
        if config.get('orth',True):
            return center_surround_orth(model_config)
        else:
            return center_surround(model_config)

    filterbank = np.cast[np.float32](filterbank) 
    return filterbank