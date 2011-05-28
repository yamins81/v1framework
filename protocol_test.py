import pythor_protocols as v1_protocols
from starflow.protocols import protocolize


@protocolize()
def image_test(depends_on = '../config_test/image_test.py'):
    v1_protocols.image_protocol(depends_on, write = True)
    

@protocolize()
def model_test(depends_on = '../config_test/model_test.py'):
    v1_protocols.model_protocol(depends_on, write = True)
    

@protocolize()
def extraction_test(depends_on = ('../config_test/model_test.py','../config_test/image_test.py')):
    v1_protocols.extract_features_protocol('../config_test/image_test.py',
                                           '../config_test/model_test.py',
                                           convolve_func_name = 'pyfft',
                                           write = True)
                                           
@protocolize()
def extraction_test_numpy(depends_on = ('../config_test/model_test.py','../config_test/image_test.py')):
    v1_protocols.extract_features_protocol('../config_test/image_test.py',
                                           '../config_test/model_test.py',
                                           convolve_func_name = 'numpy',
                                           write = True)                                           

@protocolize()                                       
def evaluation_test(depends_on=('../config_test/evaluation_test.py',
                                '../config_test/model_test.py',
                                '../config_test/image_test.py')):
    v1_protocols.evaluate_protocol('../config_test/evaluation_test.py',
                                   '../config_test/model_test.py',
                                   '../config_test/image_test.py',
                                   write=True)


@protocolize()                                       
def extract_and_evaluate_test(depends_on=('../config_test/evaluation_test.py',
                                          '../config_test/model_test.py',
                                          '../config_test/image_test.py')):
    v1_protocols.extract_and_evaluate_protocol('../config_test/evaluation_test.py',
                                   '../config_test/model_test.py',
                                   '../config_test/image_test.py',
                                   convolve_func_name='pyfft',
                                   write=True)
        

@protocolize()                                       
def extract_and_evaluate_test_parallel(depends_on=('../config_test/evaluation_test.py',
                                          '../config_test/model_test.py',
                                          '../config_test/image_test.py')):
    v1_protocols.extract_and_evaluate_protocol('../config_test/evaluation_test.py',
                                   '../config_test/model_test.py',
                                   '../config_test/image_test.py',
                                   convolve_func_name='pyfft',
                                   parallel=True,
                                   write=True)

@protocolize()                                       
def extract_and_evaluate_test_numpy(depends_on=('../config_test/evaluation_test.py',
                                          '../config_test/model_test.py',
                                          '../config_test/image_test.py')):
    v1_protocols.extract_and_evaluate_protocol('../config_test/evaluation_test.py',
                                   '../config_test/model_test.py',
                                   '../config_test/image_test.py',
                                   convolve_func_name='numpy',
                                   write=True)


@protocolize()
def parallel_image_test(depends_on = '../config_test/image_test_parallel.py'):
    v1_protocols.image_protocol(depends_on, write = True,parallel=True)

@protocolize()
def image_test_large(depends_on = '../config_test/image_test_large.py'):
    v1_protocols.image_protocol(depends_on, write = True)

@protocolize()
def image_test_large_large(depends_on = '../config_test/image_test_large_large.py'):
    v1_protocols.image_protocol(depends_on, write = True)
                

@protocolize()
def model_test_large(depends_on = '../config_test/model_test_large.py'):
    v1_protocols.model_protocol(depends_on, write = True)
     

@protocolize()
def model_test_parallel(depends_on = '../config_test/model_test_parallel.py'):
    v1_protocols.model_protocol(depends_on, write = True)

     
@protocolize()
def extraction_test_parallel(depends_on = ('../config_test/model_test_parallel.py',
                                           '../config_test/image_test_parallel.py')):
    v1_protocols.extract_features_protocol('../config_test/image_test_parallel.py',
                                           '../config_test/model_test_parallel.py',
                                           convolve_func_name = 'pyfft',
                                           write = True,
                                           parallel=True,
                                           batch_size=5)

     
@protocolize()
def extraction_test_parallel_images(depends_on = ('../config_test/model_test_parallel.py',
                                           '../config_test/image_test_parallel.py')):
    v1_protocols.extract_features_protocol('../config_test/image_test_parallel.py',
                                           '../config_test/model_test_parallel.py',
                                           convolve_func_name = 'pyfft',
                                           write = True)
                                
@protocolize()
def extraction_test_parallel_large(depends_on = ('../config_test/model_test.py',
                                           '../config_test/image_test_large.py')):
    v1_protocols.extract_features_protocol('../config_test/image_test_large.py',
                                           '../config_test/model_test.py',
                                           convolve_func_name = 'pyfft',
                                           write = True,
                                           parallel=True,
                                           batch_size=100)
        


@protocolize()
def extraction_test_parallel_large_large(depends_on = ('../config_test/model_test.py',
                                           '../config_test/image_test_large_large.py')):
    v1_protocols.extract_features_protocol('../config_test/image_test_large_large.py',
                                           '../config_test/model_test.py',
                                           convolve_func_name = 'pyfft',
                                           write = True,
                                           parallel=True,
                                           batch_size=100)
        


@protocolize()
def extraction_test_parallel_large_large_large(depends_on = ('../config_test/model_test_large.py',
                                           '../config_test/image_test_large_large.py')):
    v1_protocols.extract_features_protocol('../config_test/image_test_large_large.py',
                                           '../config_test/model_test_large.py',
                                           convolve_func_name = 'pyfft',
                                           write = True,
                                           parallel=True,
                                           batch_size=100)
        

                                           
                                   
                                           
@protocolize()                                       
def evaluation_test2(depends_on=('../config_test/evaluation_test.py',
                                 '../config_test/model_test.py',
                                 '../config_test/image_test_large.py')):
    v1_protocols.evaluate_protocol('../config_test/evaluation_test.py',
                                   '../config_test/model_test.py',
                                   '../config_test/image_test_large.py',
                                   write=True)

                                   
