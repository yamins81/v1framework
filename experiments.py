from starflow.protocols import Apply, Applies, protocolize

import deploy
import pythor_protocols as protocols


@Applies(deploy.images,args=('../config/ten_categories_images.py',True))
def ten_categories_images():
    Apply()


#@Applies(deploy.images,args=('../config/ten_categories_images_small.py',True))
#def ten_categories_images_small():
#    Apply()


@Applies(deploy.images,args=('../config/polygon_task.py',False))
def polygon_task_images():
    Apply()

@Applies(deploy.models,args=('../config/low_throughput_models.py',))
def low_throughput_models():
    Apply()
    
@protocolize()
def low_throughput_models(depends_on='../config/low_throughput_models.py'):
    protocols.model_protocol('../config/low_throughput_models.py',parallel=False,write=True)
    
@protocolize()
def extract_and_evaluate_polygons_cpu(depends_on=('../config/low_throughput_polygon.py',
                                                  '../config/low_throughput_models.py',
                                                  '../config/polygon_task.py')):
    protocols.extract_and_evaluate_protocol('../config/low_throughput_polygon.py',
                                            '../config/low_throughput_models.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='numpy', write=True,parallel=False)

@protocolize()
def extract_and_evaluate_polygons_models1_translation(depends_on=('../config/low_throughput_polygon_translation.py',
                                                  '../config/low_throughput_models.py',
                                                  '../config/polygon_task.py')):
    protocols.extract_and_evaluate_protocol('../config/low_throughput_polygon_translation.py',
                                            '../config/low_throughput_models.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='numpy', write=True,parallel=False)



@protocolize()
def low_throughput_models2(depends_on='../config/low_throughput_models2.py'):
    protocols.model_protocol('../config/low_throughput_models2.py',parallel=False,write=True)
                                            
@protocolize()
def extract_and_evaluate_polygons_cpu2(depends_on=('../config/low_throughput_polygon.py',
                                                  '../config/low_throughput_models2.py',
                                                  '../config/polygon_task.py')):
    protocols.extract_and_evaluate_protocol('../config/low_throughput_polygon.py',
                                            '../config/low_throughput_models2.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='numpy', write=True,parallel=False)

@protocolize()
def extract_and_evaluate_polygons_models2_translation(depends_on=('../config/low_throughput_polygon_translation.py',
                                                  '../config/low_throughput_models2.py',
                                                  '../config/polygon_task.py')):
    protocols.extract_and_evaluate_protocol('../config/low_throughput_polygon_translation.py',
                                            '../config/low_throughput_models2.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='numpy', write=True,parallel=False)
                                            
                                            
                                            
@protocolize()
def extract_and_evaluate_polygons_cpu_no_rotation(depends_on=('../config/low_throughput_polygon_no_rotation.py',
                                                  '../config/low_throughput_models.py',
                                                  '../config/polygon_task.py')):
    protocols.extract_and_evaluate_protocol('../config/low_throughput_polygon_no_rotation.py',
                                            '../config/low_throughput_models.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='numpy', write=True,parallel=False)

@protocolize()
def extract_and_evaluate_polygons_cpu_no_rotation2(depends_on=('../config/low_throughput_polygon_no_rotation.py',
                                                  '../config/low_throughput_models2.py',
                                                  '../config/polygon_task.py')):
    protocols.extract_and_evaluate_protocol('../config/low_throughput_polygon_no_rotation.py',
                                            '../config/low_throughput_models2.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='numpy', write=True,parallel=False)


@protocolize()
def low_throughput_models3(depends_on='../config/low_throughput_models3.py'):
    protocols.model_protocol('../config/low_throughput_models3.py',parallel=False,write=True)

@protocolize()
def extract_and_evaluate_polygons_3(depends_on=('../config/low_throughput_polygon.py',
                                                  '../config/low_throughput_models3.py',
                                                  '../config/polygon_task.py')):
    protocols.extract_and_evaluate_protocol('../config/low_throughput_polygon.py',
                                            '../config/low_throughput_models3.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='cufft', write=True,parallel=False)


@protocolize()
def extract_and_evaluate_polygons_fourier_3(depends_on=('../config/low_throughput_polygon_fourier.py',
                                                  '../config/low_throughput_models3.py',
                                                  '../config/polygon_task.py')):
    protocols.extract_and_evaluate_protocol('../config/low_throughput_polygon_fourier.py',
                                            '../config/low_throughput_models3.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='cufft', write=True,parallel=False)



@protocolize()
def low_throughput_models4(depends_on='../config/low_throughput_models4.py'):
    protocols.model_protocol('../config/low_throughput_models4.py',parallel=False,write=True)


@protocolize()
def extract_and_evaluate_polygons_fourier_4(depends_on=('../config/low_throughput_polygon_fourier.py',
                                                  '../config/low_throughput_models4.py',
                                                  '../config/polygon_task.py')):
    protocols.extract_and_evaluate_protocol('../config/low_throughput_polygon_fourier.py',
                                            '../config/low_throughput_models4.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='cufft', write=True,parallel=False)


@protocolize()
def low_throughput_models5(depends_on='../config/low_throughput_models5.py'):
    protocols.model_protocol('../config/low_throughput_models5.py',parallel=False,write=True)


@protocolize()
def extract_and_evaluate_polygons_fourier_5(depends_on=('../config/low_throughput_polygon_fourier.py',
                                                  '../config/low_throughput_models5.py',
                                                  '../config/polygon_task.py')):
    protocols.extract_and_evaluate_protocol('../config/low_throughput_polygon_fourier.py',
                                            '../config/low_throughput_models5.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='cufft', write=True,parallel=False)


@protocolize()
def extract_and_evaluate_polygons_no_rotation_translation(depends_on=('../config/low_throughput_polygon_no_rotation_translation.py',
                                                  '../config/low_throughput_models.py',
                                                  '../config/polygon_task.py')):
    protocols.extract_and_evaluate_protocol('../config/low_throughput_polygon_no_rotation_translation.py',
                                            '../config/low_throughput_models.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='numpy', write=True,parallel=False)

@protocolize()
def extract_and_evaluate_polygons_no_rotation_translation2(depends_on=('../config/low_throughput_polygon_no_rotation_translation.py',
                                                  '../config/low_throughput_models2.py',
                                                  '../config/polygon_task.py')):
    protocols.extract_and_evaluate_protocol('../config/low_throughput_polygon_no_rotation_translation.py',
                                            '../config/low_throughput_models2.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='numpy', write=True,parallel=False)



@protocolize()
def large_gabor_then_multiplication(depends_on='../config/large_gabor_then_multiplication.py'):
    protocols.model_protocol(depends_on,parallel=False,write=True)


@protocolize()
def extract_and_evaluate_large_gabor_then_multiplication_sq_vs_polygons(depends_on=('../config/sq_vs_polygon_task.py',
                                                  '../config/large_gabor_then_multiplication.py',
                                                  '../config/polygon_task.py')):
    protocols.extract_and_evaluate_protocol('../config/sq_vs_polygon_task.py',
                                            '../config/large_gabor_then_multiplication.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='cufft', write=True,parallel=False)


@protocolize()
def make_large_gabor_with_uniform_model(depends_on='../config/large_gabor_with_uniform_model.py'):
    protocols.model_protocol(depends_on,parallel=False,write=True)


@protocolize()
def extract_and_evaluate_large_gabor_with_uniform_sq_vs_polygons_translation(depends_on=('../config/sq_vs_polygon_task_translation.py',
                                                  '../config/large_gabor_with_uniform_model.py',
                                                  '../config/polygon_task.py')):
    protocols.extract_and_evaluate_protocol('../config/sq_vs_polygon_task_translation.py',
                                            '../config/large_gabor_with_uniform_model.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='numpy', write=True,parallel=False)

@protocolize()
def make_large_gabor_with_uniform_model2(depends_on='../config/large_gabor_with_uniform_model2.py'):
    protocols.model_protocol(depends_on,parallel=False,write=True)


@protocolize()
def extract_and_evaluate_large_gabor_with_uniform2_sq_vs_polygons_translation(depends_on=('../config/sq_vs_polygon_task_translation.py',
                                                  '../config/large_gabor_with_uniform_model2.py',
                                                  '../config/polygon_task.py')):
    protocols.extract_and_evaluate_protocol('../config/sq_vs_polygon_task_translation.py',
                                            '../config/large_gabor_with_uniform_model2.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='numpy', write=True,parallel=False)



@protocolize()
def medium_gabor_then_multiplication(depends_on='../config/medium_gabor_then_multiplication.py'):
    protocols.model_protocol(depends_on,parallel=False,write=True)


@protocolize()
def extract_and_evaluate_medium_gabor_then_multiplication_sq_vs_polygons(depends_on=('../config/sq_vs_polygon_task.py',
                                                  '../config/medium_gabor_then_multiplication.py',
                                                  '../config/polygon_task.py')):
    protocols.extract_and_evaluate_protocol('../config/sq_vs_polygon_task.py',
                                            '../config/medium_gabor_then_multiplication.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='cufft', write=True,parallel=False)
                                            
                                        
@protocolize()
def extract_and_evaluate_medium_gabor_then_multiplication_hex_vs_polygons(depends_on=('../config/hex_vs_polygon_task.py',
                                                  '../config/medium_gabor_then_multiplication.py',
                                                  '../config/polygon_task.py')):
    protocols.extract_and_evaluate_protocol('../config/hex_vs_polygon_task.py',
                                            '../config/medium_gabor_then_multiplication.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='cufft', write=True,parallel=False)

@protocolize()
def make_medium_gabor_with_freq_uniform_model(depends_on='../config/medium_gabor_with_freq_uniform_model.py'):
    protocols.model_protocol(depends_on,parallel=False,write=True)

@protocolize()
def extract_and_evaluate_large_gabor_then_multiplication_hex_vs_polygons(depends_on=('../config/hex_vs_polygon_task.py',
                                                  '../config/large_gabor_then_multiplication.py',
                                                  '../config/polygon_task.py')):
    protocols.extract_and_evaluate_protocol('../config/hex_vs_polygon_task.py',
                                            '../config/large_gabor_then_multiplication.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='cufft', write=True,parallel=False)

@protocolize()
def make_medium_gabor_with_freq_uniform_model(depends_on='../config/medium_gabor_with_freq_uniform_model.py'):
    protocols.model_protocol(depends_on,parallel=False,write=True)

@protocolize()
def extract_and_evaluate_medium_gabor_with_freq_uniform_sq_vs_polygons_translation(depends_on=('../config/sq_vs_polygon_task_translation.py',
                                                  '../config/medium_gabor_with_freq_uniform_model.py',
                                                  '../config/polygon_task.py')):
    protocols.extract_and_evaluate_protocol('../config/sq_vs_polygon_task_translation.py',
                                            '../config/medium_gabor_with_freq_uniform_model.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='numpy', write=True,parallel=False)
                                            


@protocolize()
def large_gabor_then_multiplication_all(depends_on='../config/large_gabor_then_multiplication_all.py'):
    protocols.model_protocol(depends_on,parallel=False,write=True)
    
@protocolize()
def extract_and_evaluate_large_gabor_then_multiplication_all_hex_vs_polygons(depends_on=('../config/hex_vs_polygon_task_small.py',
                                                  '../config/large_gabor_then_multiplication_all.py',
                                                  '../config/polygon_task.py')):
    protocols.extract_and_evaluate_protocol('../config/hex_vs_polygon_task_small.py',
                                            '../config/large_gabor_then_multiplication_all.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='cufft', write=True,parallel=False)


@protocolize()
def large_gabor(depends_on='../config/large_gabor.py'):
    protocols.model_protocol(depends_on,parallel=False,write=True)
    
@protocolize()
def extract_and_evaluate_large_gabor_hex_vs_polygons_fourier(depends_on=('../config/hex_vs_polygon_task_small_fourier.py',
                                                  '../config/large_gabor.py',
                                                  '../config/polygon_task.py')):
    protocols.extract_and_evaluate_protocol('../config/hex_vs_polygon_task_small_fourier.py',
                                            '../config/large_gabor.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='cufft', write=True,parallel=False)


@protocolize()
def larger_gabor(depends_on='../config/larger_gabor.py'):
    protocols.model_protocol(depends_on,parallel=False,write=True)
    
@protocolize()
def extract_and_evaluate_larger_gabor_hex_vs_polygons_fourier(depends_on=('../config/hex_vs_polygon_task_small_fourier.py',
                                                  '../config/larger_gabor.py',
                                                  '../config/polygon_task.py')):
    protocols.extract_and_evaluate_protocol('../config/hex_vs_polygon_task_small_fourier.py',
                                            '../config/larger_gabor.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='cufft', write=True,parallel=False)


@protocolize()
def extract_and_evaluate_larger_gabor_tri_vs_polygons_fourier(depends_on=('../config/tri_vs_polygon_task_small_fourier.py',
                                                  '../config/larger_gabor.py',
                                                  '../config/polygon_task.py')):
    protocols.extract_and_evaluate_protocol('../config/tri_vs_polygon_task_small_fourier.py',
                                            '../config/larger_gabor.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='cufft', write=True,parallel=False)



@protocolize()
def extract_and_evaluate_larger_gabor_circ_vs_polygons_fourier(depends_on=('../config/circ_vs_polygon_task_small_fourier.py',
                                                  '../config/larger_gabor.py',
                                                  '../config/polygon_task.py')):
    protocols.extract_and_evaluate_protocol('../config/circ_vs_polygon_task_small_fourier.py',
                                            '../config/larger_gabor.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='cufft', write=True,parallel=False)



@protocolize()
def more_orients_gabor(depends_on='../config/more_orients_gabor.py'):
    protocols.model_protocol(depends_on,parallel=False,write=True)
    
@protocolize()
def extract_and_evaluate_more_orients_gabor_hex_vs_polygons_fourier(depends_on=('../config/hex_vs_polygon_task_small_fourier.py',
                                                  '../config/more_orients_gabor.py',
                                                  '../config/polygon_task.py')):
    protocols.extract_and_evaluate_protocol('../config/hex_vs_polygon_task_small_fourier.py',
                                            '../config/more_orients_gabor.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='cufft', write=True,parallel=False)




    
@protocolize()
def extract_and_evaluate_large_gabor_hex_vs_polygons_fourier_no_scale(depends_on=('../config/hex_vs_polygon_task_small_fourier_no_scale.py',
                                                  '../config/large_gabor.py',
                                                  '../config/polygon_task.py')):
    protocols.extract_and_evaluate_protocol('../config/hex_vs_polygon_task_small_fourier_no_scale.py',
                                            '../config/large_gabor.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='cufft', write=True,parallel=False)
                                            
                                            
@protocolize()
def extract_and_evaluate_large_gabor_hex_vs_polygons_no_rotation(depends_on=('../config/hex_vs_polygon_task_small_no_rotation.py',
                                                  '../config/large_gabor.py',
                                                  '../config/polygon_task.py')):
    protocols.extract_and_evaluate_protocol('../config/hex_vs_polygon_task_small_no_rotation.py',
                                            '../config/large_gabor.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='cufft', write=True,parallel=False)                                            
                                            
                                            
@protocolize()
def extract_and_evaluate_large_gabor_hex_vs_polygons_no_rotation_no_translation(depends_on=('../config/hex_vs_polygon_task_small_no_rotation_no_translation.py',
                                                  '../config/large_gabor.py',
                                                  '../config/polygon_task.py')):
    protocols.extract_and_evaluate_protocol('../config/hex_vs_polygon_task_small_no_rotation_no_translation.py',
                                            '../config/large_gabor.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='cufft', write=True,parallel=False)                                                
           
           
@protocolize()
def medium_gabor(depends_on='../config/medium_gabor.py'):
    protocols.model_protocol(depends_on,parallel=False,write=True)
    
    
@protocolize()
def medium_gabor_old_norm(depends_on='../config/medium_gabor_old_norm.py'):
    protocols.model_protocol(depends_on,parallel=False,write=True)
        
    
@protocolize()
def extract_and_evaluate_medium_gabor_hex_vs_polygons_no_rotation_no_translation(depends_on=('../config/hex_vs_polygon_task_small_no_rotation_no_translation.py',
                                                  '../config/medium_gabor.py',
                                                  '../config/polygon_task.py')):
    protocols.extract_and_evaluate_protocol('../config/hex_vs_polygon_task_small_no_rotation_no_translation.py',
                                            '../config/medium_gabor.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='cufft', write=True,parallel=False)                                                                                        
                                            
                                            
@protocolize()
def extract_and_evaluate_medium_gabor_hex_vs_polygons_small(depends_on=('../config/hex_vs_polygon_task_small.py',
                                                  '../config/medium_gabor.py',
                                                  '../config/polygon_task.py')):
    protocols.extract_and_evaluate_protocol('../config/hex_vs_polygon_task_small.py',
                                            '../config/medium_gabor.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='cufft', write=True,parallel=False)                                                                                                                                    
                                            
@protocolize()
def extract_and_evaluate_medium_gabor_hex_vs_polygons_small_no_rotation(depends_on=('../config/hex_vs_polygon_task_small_no_rotation.py',
                                                  '../config/medium_gabor.py',
                                                  '../config/polygon_task.py')):
    protocols.extract_and_evaluate_protocol('../config/hex_vs_polygon_task_small_no_rotation.py',
                                            '../config/medium_gabor.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='cufft', write=True,parallel=False) 
@protocolize()
def small_gabor(depends_on='../config/small_gabor.py'):
    protocols.model_protocol(depends_on,parallel=False,write=True)
                                            

@protocolize()
def extract_and_evaluate_small_gabor_hex_vs_polygons_small_no_rotation(depends_on=('../config/hex_vs_polygon_task_small_no_rotation.py',
                                                  '../config/small_gabor.py',
                                                  '../config/polygon_task.py')):
    protocols.extract_and_evaluate_protocol('../config/hex_vs_polygon_task_small_no_rotation.py',
                                            '../config/small_gabor.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='cufft', write=True,parallel=False) 
                                            
                                            
@protocolize()
def extract_and_evaluate_medium_gabor_fourier_old_norm(depends_on=('../config/hex_vs_polygon_task_fourier.py',
                                                  '../config/medium_gabor_old_norm.py',
                                                  '../config/polygon_task.py')):
    protocols.extract_and_evaluate_protocol('../config/hex_vs_polygon_task_fourier.py',
                                            '../config/medium_gabor_old_norm.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='cufft', write=True,parallel=False) 
                                            
                
@protocolize()
def very_large_gabor(depends_on='../config/very_large_gabor.py'):
    protocols.model_protocol(depends_on,parallel=False,write=True)
    
    
@protocolize()
def extract_and_evaluate_very_large_gabor_fourier(depends_on=('../config/hex_vs_polygon_task_fourier.py',
                                                  '../config/very_large_gabor.py',
                                                  '../config/polygon_task.py')):
    protocols.extract_and_evaluate_protocol('../config/hex_vs_polygon_task_fourier.py',
                                            '../config/very_large_gabor.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='cufft', write=True,parallel=False) 
                                            
                                            
@protocolize()
def make_medium_gabor_mod_model(depends_on='../config/medium_gabor_mod.py'):
    protocols.model_protocol(depends_on,parallel=False,write=True)
    
    
@protocolize()
def extract_and_evaluate_medium_gabor_mod_fourier(depends_on=('../config/hex_vs_polygon_task_fourier.py',
                                                  '../config/medium_gabor_mod.py',
                                                  '../config/polygon_task.py')):
    protocols.extract_and_evaluate_protocol('../config/hex_vs_polygon_task_fourier.py',
                                            '../config/medium_gabor_mod.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='cufft', write=True,parallel=False) 


@protocolize()
def make_medium_gabor_mod2_model(depends_on='../config/medium_gabor_mod2.py'):
    protocols.model_protocol(depends_on,parallel=False,write=True)
    
    
@protocolize()
def extract_and_evaluate_medium_gabor_mod2_fourier(depends_on=('../config/hex_vs_polygon_task_fourier.py',
                                                  '../config/medium_gabor_mod2.py',
                                                  '../config/polygon_task.py')):
    protocols.extract_and_evaluate_protocol('../config/hex_vs_polygon_task_fourier.py',
                                            '../config/medium_gabor_mod2.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='cufft', write=True,parallel=False) 


@protocolize()
def make_medium_gabor_mod3_model(depends_on='../config/medium_gabor_mod3.py'):
    protocols.model_protocol(depends_on,parallel=False,write=True)
    
    
@protocolize()
def extract_and_evaluate_medium_gabor_mod3_fourier(depends_on=('../config/hex_vs_polygon_task_fourier.py',
                                                  '../config/medium_gabor_mod3.py',
                                                  '../config/polygon_task.py')):
    protocols.extract_and_evaluate_protocol('../config/hex_vs_polygon_task_fourier.py',
                                            '../config/medium_gabor_mod3.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='cufft', write=True,parallel=False) 


@protocolize()
def make_medium_gabor_mod4_model(depends_on='../config/medium_gabor_mod4.py'):
    protocols.model_protocol(depends_on,parallel=False,write=True)
    
    
@protocolize()
def extract_and_evaluate_medium_gabor_mod4_fourier(depends_on=('../config/hex_vs_polygon_task_fourier2.py',
                                                  '../config/medium_gabor_mod4.py',
                                                  '../config/polygon_task.py')):
    protocols.extract_and_evaluate_protocol('../config/hex_vs_polygon_task_fourier2.py',
                                            '../config/medium_gabor_mod4.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='cufft', write=True,parallel=False) 

@protocolize()
def extract_and_evaluate_medium_gabor_mod4_fourier2(depends_on=('../config/various_vs_polygon_task2.py',
                                                  '../config/medium_gabor_mod4.py',
                                                  '../config/polygon_task.py')):
    protocols.extract_and_evaluate_protocol('../config/various_vs_polygon_task2.py',
                                            '../config/medium_gabor_mod4.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='cufft', write=True,parallel=False) 

@protocolize()
def make_medium_gabor_mod5_model(depends_on='../config/medium_gabor_mod5.py'):
    protocols.model_protocol(depends_on,parallel=False,write=True)
    
    
@protocolize()
def extract_and_evaluate_medium_gabor_mod5_fourier(depends_on=('../config/hex_vs_polygon_task_fourier2.py',
                                                  '../config/medium_gabor_mod5.py',
                                                  '../config/polygon_task.py')):
    protocols.extract_and_evaluate_protocol('../config/hex_vs_polygon_task_fourier2.py',
                                            '../config/medium_gabor_mod5.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='cufft', write=True,parallel=False) 


@protocolize()
def make_medium_gabor_mod4_with_multiplication_model(depends_on='../config/medium_gabor_mod4_with_multiplication.py'):
    protocols.model_protocol(depends_on,parallel=False,write=True)
    
    
@protocolize()
def extract_and_evaluate_medium_gabor_mod4_with_multiplication(depends_on=('../config/various_vs_polygon_task.py',
                                                  '../config/medium_gabor_mod4_with_multiplication.py',
                                                  '../config/polygon_task.py')):
    protocols.extract_and_evaluate_protocol('../config/various_vs_polygon_task.py',
                                            '../config/medium_gabor_mod4_with_multiplication.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='cufft', write=True,parallel=False) 


@protocolize()
def make_medium_gabor_mod4a_model(depends_on='../config/medium_gabor_mod4a.py'):
    protocols.model_protocol(depends_on,parallel=False,write=True)
    

@protocolize()
def extract_and_evaluate_medium_gabor_mod4a_fourier2(depends_on=('../config/various_vs_polygon_task2.py',
                                                  '../config/medium_gabor_mod4a.py',
                                                  '../config/polygon_task.py')):
    protocols.extract_and_evaluate_protocol('../config/various_vs_polygon_task2.py',
                                            '../config/medium_gabor_mod4a.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='cufft', write=True,parallel=False) 


@protocolize()
def make_medium_gabor_mod4_with_multiplication_all_model(depends_on='../config/medium_gabor_mod4_with_multiplication_all.py'):
    protocols.model_protocol(depends_on,parallel=False,write=True)
    

@protocolize()
def extract_and_evaluate_medium_gabor_mod4_with_multiplication_all(depends_on=('../config/various_vs_polygon_task3.py',
                                                  '../config/medium_gabor_mod4_with_multiplication_all.py',
                                                  '../config/polygon_task.py')):
    protocols.extract_and_evaluate_protocol('../config/various_vs_polygon_task3.py',
                                            '../config/medium_gabor_mod4_with_multiplication_all.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='cufft', write=True,parallel=False) 

@protocolize()
def make_small_gabor_with_multiplication_model(depends_on='../config/small_gabor_with_multiplication.py'):
    protocols.model_protocol(depends_on,parallel=False,write=True)
    
    
@protocolize()
def extract_and_evaluate_small_gabor_with_multiplication(depends_on=('../config/various_vs_polygon_task4.py',
                                                  '../config/small_gabor_with_multiplication.py',
                                                  '../config/polygon_task.py')):
    protocols.extract_and_evaluate_protocol('../config/various_vs_polygon_task4.py',
                                            '../config/small_gabor_with_multiplication.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='cufft', write=True,parallel=False) 
                                            
@protocolize()
def make_small_gabor_with_multiplication2_model(depends_on='../config/small_gabor_with_multiplication2.py'):
    protocols.model_protocol(depends_on,parallel=False,write=True)
    
    
@protocolize()
def extract_and_evaluate_small_gabor_with_multiplication2(depends_on=('../config/various_vs_polygon_task4.py',
                                                  '../config/small_gabor_with_multiplication2.py',
                                                  '../config/polygon_task.py')):
    protocols.extract_and_evaluate_protocol('../config/various_vs_polygon_task4.py',
                                            '../config/small_gabor_with_multiplication2.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='cufft', write=True,parallel=False) 
                                            
@protocolize()
def make_small_gabor_with_uniform_model(depends_on='../config/small_gabor_with_uniform.py'):
    protocols.model_protocol(depends_on,parallel=False,write=True)
    
    
@protocolize()
def extract_and_evaluate_small_gabor_with_uniform(depends_on=('../config/various_vs_polygon_task5.py',
                                                  '../config/small_gabor_with_uniform.py',
                                                  '../config/polygon_task.py')):
    protocols.extract_and_evaluate_protocol('../config/various_vs_polygon_task5.py',
                                            '../config/small_gabor_with_uniform.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='numpy', write=True,parallel=False) 
                                            
                                            
@protocolize()
def extract_and_evaluate_small_gabor_with_uniform_sumup(depends_on=('../config/various_vs_polygon_task6.py',
                                                  '../config/small_gabor_with_uniform.py',
                                                  '../config/polygon_task.py')):
    protocols.extract_and_evaluate_protocol('../config/various_vs_polygon_task6.py',
                                            '../config/small_gabor_with_uniform.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='numpy', write=True,parallel=False) 
                                            
@protocolize()
def make_medium_gabor_mod4_with_freq_uniform_model(depends_on='../config/medium_gabor_mod4_with_freq_uniform.py'):
    protocols.model_protocol(depends_on,parallel=False,write=True)

    
@protocolize()
def extract_and_evaluate_medium_gabor_mod4_with_freq_uniform_sumup(depends_on=('../config/various_vs_polygon_task6.py',
                                                  '../config/medium_gabor_mod4_with_freq_uniform.py',
                                                  '../config/polygon_task.py')):
    protocols.extract_and_evaluate_protocol('../config/various_vs_polygon_task6.py',
                                            '../config/medium_gabor_mod4_with_freq_uniform.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='numpy', write=True,parallel=False)
                                            
@protocolize()
def ext_eval_medium_gabor_mod4_with_freq_uniform_ten_categories(depends_on=('../config/ten_categories_renderman_task.py',
                                                  '../config/medium_gabor_mod4_with_freq_uniform.py',
                                                  '../config/ten_categories_images.py')):
    protocols.extract_and_evaluate_protocol('../config/ten_categories_renderman_task.py',
                                            '../config/medium_gabor_mod4_with_freq_uniform.py',
                                            '../config/ten_categories_images.py',
                                            convolve_func_name='numpy', write=True,parallel=False)
                                            
                                            
@protocolize()
def make_best_l3_model(depends_on='../config/best_l3_model.py'):
    protocols.model_protocol(depends_on,parallel=False,write=True)

                                            
@protocolize()
def ext_eval_best_l3_ten_categories(depends_on=('../config/ten_categories_renderman_task2.py',
                                                  '../config/best_l3_model.py',
                                                  '../config/ten_categories_images.py')):
    protocols.extract_and_evaluate_protocol('../config/ten_categories_renderman_task2.py',
                                            '../config/best_l3_model.py',
                                            '../config/ten_categories_images.py',
                                            convolve_func_name='numpy', write=True,parallel=False)
                                            
@protocolize()
def ext_eval_best_l3_polygon(depends_on=('../config/various_vs_polygon_task0.py',
                                                  '../config/best_l3_model.py',
                                                  '../config/polygon_task.py')):
    protocols.extract_and_evaluate_protocol('../config/various_vs_polygon_task0.py',
                                            '../config/best_l3_model.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='numpy', write=True,parallel=False)
                                            
                                            
@protocolize()
def make_medium_l1_random_model(depends_on='../config/l1_random_120.py'):
    protocols.model_protocol(depends_on,parallel=False,write=True)
    
@protocolize()
def extract_and_evaluate_medium_l1_random(depends_on=('../config/various_vs_polygon_task8.py',
                                                  '../config/l1_random_120.py',
                                                  '../config/polygon_task.py')):
    protocols.extract_and_evaluate_protocol('../config/various_vs_polygon_task8.py',
                                            '../config/l1_random_120.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='cufft', write=True,parallel=False)                                        
                                            
                                            
@protocolize()
def make_medium_random_multiply_model(depends_on='../config/random_multiply.py'):
    protocols.model_protocol(depends_on,parallel=False,write=True)
    
@protocolize()
def extract_and_evaluate_random_multiply(depends_on=('../config/various_vs_polygon_task7.py',
                                                  '../config/random_multiply.py',
                                                  '../config/polygon_task.py')):
    protocols.extract_and_evaluate_protocol('../config/various_vs_polygon_task7.py',
                                            '../config/random_multiply.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='cufft', write=True,parallel=False)   
                                            
@protocolize()
def make_medium_random_multiply_max_model(depends_on='../config/random_multiply_max.py'):
    protocols.model_protocol(depends_on,parallel=False,write=True)
    
@protocolize()
def extract_and_evaluate_random_multiply_max(depends_on=('../config/various_vs_polygon_task7.py',
                                                  '../config/random_multiply_max.py',
                                                  '../config/polygon_task.py')):
    protocols.extract_and_evaluate_protocol('../config/various_vs_polygon_task7.py',
                                            '../config/random_multiply_max.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='cufft', write=True,parallel=False)    
                                            
                                            
                                            
@protocolize()
def make_medium_gabor_mod4_with_multiplication_no_sum_model(depends_on='../config/medium_gabor_mod4_with_multiplication_no_sum.py'):
    protocols.model_protocol(depends_on,parallel=False,write=True)
    
    
@protocolize()
def extract_and_evaluate_medium_gabor_mod4_with_multiplication_no_sum(depends_on=('../config/various_vs_polygon_task7.py',
                                                  '../config/medium_gabor_mod4_with_multiplication_no_sum.py',
                                                  '../config/polygon_task.py')):
    protocols.extract_and_evaluate_protocol('../config/various_vs_polygon_task7.py',
                                            '../config/medium_gabor_mod4_with_multiplication_no_sum.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='cufft', write=True,parallel=False) 

@protocolize()
def ext_eval_medium_gabor_mod4_mult_ten_categories(depends_on=('../config/ten_categories_renderman_task3.py',
                                                  '../config/medium_gabor_mod4_with_multiplication_no_sum.py',
                                                  '../config/ten_categories_images.py')):
    protocols.extract_and_evaluate_protocol('../config/ten_categories_renderman_task3.py',
                                            '../config/medium_gabor_mod4_with_multiplication_no_sum.py',
                                           '../config/ten_categories_images.py',
                                            convolve_func_name='numpy', write=True,parallel=False) 
