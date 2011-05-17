from pycuda.tools import make_default_context
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import numpy as np

try:
    from pyfft.cuda import Plan as pyfft_Plan
except:
    print('pyfft unavailable')
else:
    print('pyfft available')
    
try:
    import scikits.cuda.fft as cu_fft
except:
    print('cufft unavailable')
else:
    print('cufft available')
    

CONTEXTS = {}
PYFFT_PLANS = {}

def setup_pyfft(device_id=0):
    global CONTEXTS
    if device_id not in CONTEXTS:
        print('initializing GPU device', device_id)
        cuda.init()
        if device_id is None:
            CONTEXTS[0] = make_default_context()
        else:
            dev = cuda.Device(device_id)
            CONTEXTS[device_id] = dev.make_context()
            CONTEXTS[device_id].pop()
    return CONTEXTS[device_id]
        


def cleanup_pyfft(device_id):
    pass

def pad(data,shape):

    newdata = np.zeros(shape,dtype=data.dtype)      
    
    sel1 = []
    sel2 = []
         
    for (d,s) in zip(data.shape,shape):
        if s > d:    
            sel1.append ( slice((s-d)/2 , (s-d)/2 + d) )  
            sel2.append ( slice(0, d) )
        else:
            sel1.append ( slice(0, s) )
            sel2.append ( slice((d-s)/2 , (d-s)/2 + s))
    
    newdata[tuple(sel1)] = data[sel2]
    
    return newdata

def pad2(data,shape):

    newdata = np.zeros(shape,dtype=data.dtype)      
    
    sel1 = []
    sel2 = []
         
    for (d,s) in zip(data.shape,shape):
        if s > d:    
            sel1.append ( slice(0, d) )  
            sel2.append ( slice(0, d) )
        else:
            sel1.append ( slice(0, s) )
            sel2.append ( slice(0, s) )
    
    newdata[tuple(sel1)] = data[sel2]
    
    return newdata    


def fft(data,shape=None,inverse=False,device_id=0):

    if shape:
        data = pad2(data,shape)
                        
    plan = PYFFT_PLANS.get((device_id,data.shape))
    if not plan:
        plan = pyfft_Plan(data.shape)
        PYFFT_PLANS[(device_id,data.shape)] = plan
    
    gpu_data = gpuarray.to_gpu(np.cast[np.complex64](data))
    plan.execute(gpu_data,inverse = inverse)
    r = gpu_data.get()
    
    return r


def cufft(data,shape=None,inverse=False):

    if shape:
        data = pad2(data,shape)
                        
    plan  = CUFFT_PLANS.get(data.shape)
    if not plan:
        plan = cu_fft.Plan(data.shape,np.complex64,np.complex64)
        CUFFT_PLANS[data.shape] = plan
    
    gpu_data = gpuarray.to_gpu(np.cast[np.complex64](data))
    if inverse:
        cu_fft.ifft(gpu_data,gpu_data,plan)
    else:
        cu_fft.fft(gpu_data,gpu_data,plan)
    r = gpu_data.get()
    
    return r
