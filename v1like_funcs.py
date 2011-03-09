#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" v1like_funcs module

Key sub-operations performed in a simple V1-like model 
(normalization, linear filtering, downsampling, etc.)

"""
import time
import scipy as sp
import scipy.signal

try:
    import v1_pyfft
except:
    pass

from npclockit import clockit_onprofile

conv = scipy.signal.convolve

PROFILE = False

# -------------------------------------------------------------------------
@clockit_onprofile(PROFILE)
def v1s_norm(hin, conv_mode, kshape, threshold):
    """ V1S local normalization
    
    Each pixel in the input image is divisively normalized by the L2 norm
    of the pixels in a local neighborhood around it, and the result of this
    division is placed in the output image.   
    
    Inputs:
      hin -- a 3-dimensional array (width X height X rgb)
      kshape -- kernel shape (tuple) ex: (3,3) for a 3x3 normalization 
                neighborhood
      threshold -- magnitude threshold, if the vector's length is below 
                   it doesn't get resized ex: 1.    
     
    Outputs:
      hout -- a normalized 3-dimensional array (width X height X rgb)
      
    """
    
    eps = 1e-5
    kh, kw = kshape
    dtype = hin.dtype
    hsrc = hin[:].copy()

    # -- prepare hout
    hin_h, hin_w, hin_d = hin.shape
    hout_h = hin_h - kh + 1
    hout_w = hin_w - kw + 1
    hout_d = hin_d    
    hout = sp.empty((hout_h, hout_w, hout_d), 'f')

    # -- compute numerator (hnum) and divisor (hdiv)
    # sum kernel
    hin_d = hin.shape[-1]
    kshape3d = list(kshape) + [hin_d]            
    ker = sp.ones(kshape3d, dtype=dtype)
    size = ker.size

    # compute sum-of-square
    hsq = hsrc ** 2.
    hssq = conv(hsq, ker, conv_mode).astype(dtype)

    # compute hnum and hdiv
    ys = kh / 2
    xs = kw / 2
    hout_h, hout_w, hout_d = hout.shape[-3:]
    hs = hout_h
    ws = hout_w
    hsum = conv(hsrc, ker, conv_mode).astype(dtype)
    hnum = hsrc[ys:ys+hs, xs:xs+ws] - (hsum/size)
    val = (hssq - (hsum**2.)/size)
    sp.putmask(val, val<0, 0) # to avoid negative sqrt
    hdiv = val ** (1./2) + eps

    # -- apply normalization
    # 'volume' threshold
    sp.putmask(hdiv, hdiv < (threshold+eps), 1.)
    result = (hnum / hdiv)
    
    hout[:] = result
    return hout


@clockit_onprofile(PROFILE)
def v1like_norm(hin, conv_mode, kshape, threshold):
    """ V1LIKE local normalization
    
    Each pixel in the input image is divisively normalized by the L2 norm
    of the pixels in a local neighborhood around it, and the result of this
    division is placed in the output image.   
    
    Inputs:
      hin -- a 3-dimensional array (width X height X rgb)
      kshape -- kernel shape (tuple) ex: (3,3) for a 3x3 normalization 
                neighborhood
      threshold -- magnitude threshold, if the vector's length is below 
                   it doesn't get resized ex: 1.    
     
    Outputs:
      hout -- a normalized 3-dimensional array (width X height X rgb)
      
    """
    
    eps = 1e-5
    kh, kw = kshape
    dtype = hin.dtype
    hsrc = hin[:].copy()

    # -- prepare hout
    hin_h, hin_w, hin_d = hin.shape
    hout_h = hin_h# - kh + 1
    hout_w = hin_w# - kw + 1

    if conv_mode != "same":
        hout_h = hout_h - kh + 1
        hout_w = hout_w - kw + 1
        
    hout_d = hin_d    
    hout = sp.empty((hout_h, hout_w, hout_d), 'float32')

    # -- compute numerator (hnum) and divisor (hdiv)
    # sum kernel
    hin_d = hin.shape[-1]
    kshape3d = list(kshape) + [hin_d]            
    ker = sp.ones(kshape3d, dtype=dtype)
    size = ker.size

    # compute sum-of-square
    hsq = hsrc ** 2.
    #hssq = conv(hsq, ker, conv_mode).astype(dtype)
    kerH = ker[:,0,0][:, None]#, None]
    kerW = ker[0,:,0][None, :]#, None]
    kerD = ker[0,0,:][None, None, :]

    #s = time.time()
    #r = conv(hsq, kerD, 'valid')[:,:,0]
    #print time.time()-s

    #s = time.time()
    hssq = conv(kerH, conv(kerW, conv(hsq, kerD, 'valid')[:,:,0].astype(dtype), conv_mode), conv_mode).astype(dtype)
    hssq = hssq[:,:,None]
    #print time.time()-s

    # compute hnum and hdiv
    ys = kh / 2
    xs = kw / 2
    hout_h, hout_w, hout_d = hout.shape[-3:]
    hs = hout_h
    ws = hout_w
    #hsum = conv(hsrc, ker, conv_mode).astype(dtype)
    hsum = conv(kerH, 
                conv(kerW, 
                     conv(hsrc, 
                          kerD, 'valid')[:,:,0].astype(dtype), 
                     conv_mode),
                conv_mode).astype(dtype)
    hsum = hsum[:,:,None]
    if conv_mode == 'same':
        hnum = hsrc - (hsum/size)
    else:
        hnum = hsrc[ys:ys+hs, xs:xs+ws] - (hsum/size)
    val = (hssq - (hsum**2.)/size)
    val[val<0] = 0
    hdiv = val ** (1./2) + eps

    # -- apply normalization
    # 'volume' threshold
    sp.putmask(hdiv, hdiv < (threshold+eps), 1.)
    result = (hnum / hdiv)
    
    #print result.shape
    hout[:] = result
    #print hout.shape, hout.dtype
    return hout

# -------------------------------------------------------------------------
FILTER_FFT_CACHE = {}

def power2(shape):
    return tuple([int(math.ceil(math.log(x,2))) for x in shape])


def get_bounds(image_shape,filter_shape,conv_mode):
    if conv_mode == "valid":        
        out_shape = tuple( np.array(image_shape[:2]) - np.array(filter_shape[:2]) + 1 )
        begy = filter_shape[0]
        endy = begy + out_shape[0]
        begx = filter_shape[1]
        endx = begx + out_shape[1]
    elif conv_mode == "same":
        out_shape = image_shape[:2] 
        begy = filter_shape[0] / 2
        endy = begy + out_shape[0]
        begx = filter_shape[1] / 2
        endx = begx + out_shape[1]
    elif conv_mode == "full":
        out_shape = tuple( np.array(image_shape[:2]) + np.array(filter_shape[:2]) - 1 )
        begx = 0
        endx = out_shape[0]
        begx = 0
        endy = out_shape[1]
    else:
        raise NotImplementedError 
        
    return (slice(begx,endx),slice(begy,endy))    


def v1like_filter_numpy(img_source,filter_source,image_config,filter_config):

    conv_mode = image_config['conv_mode']
    filter_shape = filter_config['filter']['kshape']
    
    image = img_source()
    image_shape = image.shape
    image_fft = np.fft.fftn(image,fft_shape)
    
    fft_shape = tuple( np.array(image_shape) + np.array(filter_shape) - 1 )    

    filter_key = (fft_shape,filter_config.items())
        
    if filter_key not in FILTER_FFT_CACHE:
        filterbank = filter_source()
        filter_fft = np.empty(fft_shape + (filterbank.shape[2],),dtype=filterbank.dtype)
        for i in range(filterbank.shape[2]):
            filter_fft[:,:,i] = np.fft.fftn(filterbank[:,:,i],fft_shape)
        FILTER_FFT_CACHE[filter_key] = filter_fft
    else:
        filter_fft = FILTER_FFT_CACHE[filter_key]
    
    res_fft = np.empty(filter_fft.shape,dtype=filter_fft.dtype)
    for i in range(res_fft.shape[2]):
        res_fft[:,:,i] = np.fft.iffn(image_fft * filter_fft[:,:,i])
        
    myslice = get_bounds(image_shape,filter_shape,conv_mode)

    res_fft = res_fft[myslice]
        
    return res_fft
    
    
def v1like_filter_pyfft(img_source,filter_source,image_config,filter_config):

    conv_mode = image_config['conv_mode']
    filter_shape = filter_config['filter']['kshape']
    
    image = img_source()
    image_shape = image.shape
    image_fft = np.fft.fftn(image,fft_shape)

    
    full_shape = tuple( np.array(image_shape) + np.array(filter_shape) - 1 )
    fft_shape = power2(full_shape) 

    filter_key = (fft_shape,filter_config.items())
        
    if filter_key not in FILTER_FFT_CACHE:
        filterbank = filter_source()
        filter_fft = np.empty(fft_shape + (filterbank.shape[2],),dtype=filterbank.dtype)
        for i in range(filterbank.shape[2]):
            filter_fft[:,:,i] = v1_pyfft.fft(filterbank[:,:,i],fft_shape)
        FILTER_FFT_CACHE[filter_key] = filter_fft
    else:
        filter_fft = FILTER_FFT_CACHE[filter_key]
    
    res_fft = np.empty(filter_fft.shape,dtype=filter_fft.dtype)
    for i in range(res_fft.shape[2]):
        res_fft[:,:,i] = v1_pyfft.fft(image_fft * filter_fft[:,:,i],reverse=True)
        
    myslice = tuple([slice(d,d + f) for (d,f) in zip(delta,full_shape)])

    res_fft = res_fft[myslice]
    
    myslice = get_bounds(image_shape,filter_shape,conv_mode)           
    res_fft = res_fft[myslice]
    
    return res_fft   
    
    
# -------------------------------------------------------------------------

@clockit_onprofile(PROFILE)
#@profile
def v1like_pool(hin, conv_mode, lsum_ksize, outshape=None, order=1):
    """ V1LIKE Pooling
    Boxcar Low-pass filter featuremap-wise
    
    Inputs:
      hin -- a 3-dimensional array (width X height X n_channels)
      lsum_ksize -- kernel size of the local sum ex: 17
      outshape -- fixed output shape (2d slices)
      order -- XXX
     
    Outputs:
       hout -- resulting 3-dimensional array

    """

    order = float(order)
    assert(order >= 1)
    
    # -- local sum
    if lsum_ksize is not None:
        hin_h, hin_w, hin_d = hin.shape
        dtype = hin.dtype
        if conv_mode == "valid":
            aux_shape = auxh, auxw, auxd = hin_h-lsum_ksize+1, hin_w-lsum_ksize+1, hin_d
            aux = sp.empty(aux_shape, dtype)
        else:
            aux = sp.empty(hin.shape, dtype)
        k1d = sp.ones((lsum_ksize), 'f')
        k2d = sp.ones((lsum_ksize, lsum_ksize), 'f')
        krow = k1d[None,:]
        kcol = k1d[:,None]
        for d in xrange(aux.shape[2]):
            if order == 1:
                aux[:,:,d] = conv(conv(hin[:,:,d], krow, conv_mode), kcol, conv_mode)
            else:
                aux[:,:,d] = conv(conv(hin[:,:,d]**order, krow, conv_mode), kcol, conv_mode)**(1./order)
                
    else:
        aux = hin

    # -- resample output
    if outshape is None or outshape == aux.shape:
        hout = aux
    else:
        hout = sresample(aux, outshape)
        
    return hout

# -------------------------------------------------------------------------
@clockit_onprofile(PROFILE)
def sresample(src, outshape):
    """ Simple 3d array resampling

    Inputs:
      src -- a ndimensional array (dim>2)
      outshape -- fixed output shape for the first 2 dimensions
     
    Outputs:
       hout -- resulting n-dimensional array
    
    """
    
    inh, inw = inshape = src.shape[:2]
    outh, outw = outshape
    hslice = (sp.arange(outh) * (inh-1.)/(outh-1.)).round().astype(int)
    wslice = (sp.arange(outw) * (inw-1.)/(outw-1.)).round().astype(int)
    hout = src[hslice, :][:, wslice]    
    return hout.copy()
    
    