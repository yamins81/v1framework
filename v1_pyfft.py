from pyfft.cuda import Plan
from pycuda.tools import make_default_context
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import numpy as np

CONTEXT = None
PLANS = {}

def setup_pyfft():
    cuda.init()
    global CONTEXT
    CONTEXT = make_default_context()

    
def cleanup_pyfft():
    CONTEXT.pop()


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

def fft(data,shape=None,inverse=False):

    if shape:
        data = pad(data,shape)
                        
    plan  = PLANS.get(data.shape)
    if not plan:
        plan = Plan(data.shape)
        PLANS[data.shape] = plan
    
    gpu_data = gpuarray.to_gpu(np.cast[np.complex64](data))
    plan.execute(gpu_data,inverse = inverse)
    r = gpu_data.get()
    
    return r

