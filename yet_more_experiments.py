from starflow.protocols import Apply, Applies, protocolize

import deploy
import pythor_protocols as protocols


@Applies(deploy.images,args=('../config/faces_images.py',True))
def generate_faces_images():
    Apply()

@protocolize()
def ext_eval_various_random_l3_face_subtasks(depends_on=('../config/face_subtasks.py',
                                                  '../config/various_random_l3_grayscale_models.py',
                                                  '../config/faces_images.py')):
    """
    all: (21.269841269841269, 13.333333333333334, 16.884920634920636, 2.6135554487418613)
    just trans: (100.0, 98.412698412698418, 99.642857142857153, 0.50037778622692353)
    in-plane: (96.19047619047619, 82.222222222222214, 90.813492063492049, 3.8086453186818607)
    scale; (98.412698412698418, 89.523809523809518, 95.416666666666657, 2.4045621741142726)
    inplane and scale: (52.04081632653061, 39.455782312925173, 46.301020408163268, 3.5853208104055021)
    """
    protocols.extract_and_evaluate_protocol('../config/face_subtasks.py',
                                            '../config/various_random_l3_grayscale_models.py',
                                            '../config/faces_images.py',
                                            convolve_func_name='numpy', write=True,parallel=True)



###############


@protocolize()
def ext_eval_ht_l2_random_random_renderman(depends_on=('../config/renderman_tasks_for_ht2.py',
                                                  '../config/ht_l2_random_random_models.py',
                                                  '../config/ten_categories_images.py')):
    """
    overall: 42.67, 24.53, 33.35, 36.70, 4.38              
    
    level 1, lnorm, inker_shape
    value: [3, 5, 9]
    max: 41.73, 42.13, 42.67
    mean: 33.49, 33.17, 33.40
    
    level 1, lpool, ker_shape
    value: [3, 5, 9]
    max: 36.13, 42.67, 42.13
    mean: 29.23, 33.22, 37.61
    
    level 1, filter, ker_shape
    value: [5, 7, 9, 11, 13, 17]
    max: 42.67, 41.47, 40.40, 41.73, 40.40, 37.33
    mean: 36.19, 34.93, 33.64, 32.99, 32.04, 30.33
    
    level 2, filter, ker_shape
    value: [3, 5, 7, 9]
    max: 41.20, 41.20, 42.13, 42.67
    mean: 32.26, 33.49, 33.37, 34.29
    
    """
    protocols.extract_and_evaluate_protocol('../config/renderman_tasks_for_ht2.py',
                                            '../config/ht_l2_random_random_models.py',
                                            '../config/ten_categories_images.py',
                                            convolve_func_name='numpy', write=True,parallel=True)

@protocolize()
def ext_eval_ht_l2_gabor_random_renderman(depends_on=('../config/renderman_tasks_for_ht2.py',
                                                  '../config/ht_l2_gabor_random_models.py',
                                                  '../config/ten_categories_images.py')):
    """
    overall: 47.47, 29.07, 38.21, 40.30, 3.23
    with 30/256 filters: 43.07, 31.33, 38.53, 40.20, 2.60
    
	level 1, lnorm, inker_shape
	value: [3, 5]
	max: 47.47, 45.20
	mean: 38.43, 37.99
	
	level 1, filter, norients
	value: [4, 6, 8]
	max: 44.40, 47.47, 45.20
	mean: 37.85, 38.53, 38.25
	
	level 1, filter, ker_shape
	value: [5, 7, 9, 11, 13, 17]
	max: 45.20, 44.40, 44.13, 44.13, 47.47, 39.73
	mean: 39.83, 39.29, 39.28, 38.39, 37.69, 34.79
	
	level 2, filter, ker_shape
	value: [3, 5]
	max: 44.67, 47.47
	mean: 37.51, 38.91
	
	level 2, filter, num_filters
	value: [128, 256, 384]
	max: 40.80, 43.60, 47.47
	mean: 35.82, 38.59, 40.22
    
    """
    protocols.extract_and_evaluate_protocol('../config/renderman_tasks_for_ht2.py',
                                            '../config/ht_l2_gabor_random_models.py',
                                            '../config/ten_categories_images.py',
                                            convolve_func_name='numpy', write=True,parallel=True)


@protocolize()
def ext_eval_ht_l2_freq_renderman(depends_on=('../config/renderman_tasks_for_ht2.py',
                                                  '../config/ht_l2_freq_models.py',
                                                  '../config/ten_categories_images.py')):
    """
    
    """
    protocols.extract_and_evaluate_protocol('../config/renderman_tasks_for_ht2.py',
                                            '../config/ht_l2_freq_models.py',
                                            '../config/ten_categories_images.py',
                                            convolve_func_name='numpy', write=True,parallel=True)


@protocolize()
def make_ht_l1_gabor_models(depends_on='../config/ht_l1_gabor_models_more.py'):
    """
  
    """
    protocols.model_protocol(depends_on,parallel=False,write=True)


@protocolize()
def ext_eval_ht_l1_gabor_renderman(depends_on=('../config/renderman_tasks_for_ht2.py',
                                                  '../config/ht_l1_gabor_models_more.py',
                                                  '../config/ten_categories_images.py')):
    """
    overall: (37.866666666666667, 10.0, 26.661728395061733, 8.6711750083247772)

    level 0, lnorm, inker_shape
    value: [3, 9]
    max: 37.87, 37.20
    mean: 27.99, 25.33
        
    level 1, lnorm, inker_shape
    value: [3, 7]
    max: 37.20, 37.87
    mean: 26.77, 26.5
    
    level 1, filter, ker_shape
    value: [9, 11, 13, 17, 21, 31]
    max: 37.20, 37.87, 36.13, 34.93, 34.27, 32.13
    mean: 27.87, 27.65, 27.25, 26.77, 25.67, 24.75
    
    level 1, activ, min_out
    value: [-0.5, 0, 0.5]
    max: 37.20, 37.87, 21.47
    mean: 31.98, 32.81, 15.19
    
    level 1, lpool, ker_shape
    value: [5, 9, 13]
    max: 36.93, 37.20, 37.87
    mean: 26.56, 26.74, 26.68

    """
    protocols.extract_and_evaluate_protocol('../config/renderman_tasks_for_ht2.py',
                                            '../config/ht_l1_gabor_models_more.py',
                                            '../config/ten_categories_images.py',
                                            convolve_func_name='numpy', write=True,parallel=True)


@protocolize()
def ext_eval_ht_l1_random_renderman(depends_on=('../config/renderman_tasks_for_ht2.py',
                                                  '../config/ht_l1_random_models.py',
                                                  '../config/ten_categories_images.py')):
    """
    
    """
    protocols.extract_and_evaluate_protocol('../config/renderman_tasks_for_ht2.py',
                                            '../config/ht_l1_random_models.py',
                                            '../config/ten_categories_images.py',
                                            convolve_func_name='numpy', write=True,parallel=True)
                                            
                                            
###################                                           
@protocolize()
def make_l2_gabor_corr_test_model(depends_on='../config/l2_gabor_corr_test_model.py'):
    """
    """
    protocols.model_protocol(depends_on,parallel=False,write=True)


@protocolize()
def ext_eval_l2_gabor_corr_test_renderman(depends_on=('../config/renderman_tasks_for_ht2.py',
                                                  '../config/l2_gabor_corr_test_model.py',
                                                  '../config/ten_categories_images.py')):
    """
    no different than random 
    """
    protocols.extract_and_evaluate_protocol('../config/renderman_tasks_for_ht2.py',
                                            '../config/l2_gabor_corr_test_model.py',
                                            '../config/ten_categories_images.py',
                                            convolve_func_name='numpy', write=True,parallel=True)

@protocolize()
def ext_eval_l2_gabor_corr_test_renderman_specific(depends_on=('../config/renderman_tasks_for_corr_test.py',
                                                  '../config/l2_gabor_corr_test_model.py',
                                                  '../config/ten_categories_images.py')):
    """
     seeing if the corr method does better than rando on the restriccted problem of just rotation
     result: it doesn't
    """
    protocols.extract_and_evaluate_protocol('../config/renderman_tasks_for_corr_test.py',
                                            '../config/l2_gabor_corr_test_model.py',
                                            '../config/ten_categories_images.py',
                                            convolve_func_name='numpy', write=True,parallel=True)


@protocolize()
def make_l2_gabor_corr_test_model_2(depends_on='../config/l2_gabor_corr_test_model2.py'):
    """
    trying corr again with different underlying l1 gabor model
    """
    protocols.model_protocol(depends_on,parallel=False,write=True)


@protocolize()
def ext_eval_l2_gabor_corr_test_2_renderman(depends_on=('../config/renderman_tasks_for_ht2.py',
                                                  '../config/l2_gabor_corr_test_model2.py',
                                                  '../config/ten_categories_images.py')):
    """
    again, no better than random
    """
    protocols.extract_and_evaluate_protocol('../config/renderman_tasks_for_ht2.py',
                                            '../config/l2_gabor_corr_test_model2.py',
                                            '../config/ten_categories_images.py',
                                            convolve_func_name='numpy', write=True,parallel=True)
                                            


@protocolize()
def make_l2_gabor_random_corr_test_comparison_model(depends_on='../config/l2_gabor_random_corr_test_comparison_model.py'):
    """
    """
    protocols.model_protocol(depends_on,parallel=False,write=True)


@protocolize()
def ext_eval_l2_gabor_random_corr_test_comparison_renderman(depends_on=('../config/renderman_tasks_for_ht2.py',
                                                  '../config/l2_gabor_random_corr_test_comparison_model.py',
                                                  '../config/ten_categories_images.py')):
    """
    the thing against which to compare corr test above - does equally well
    """
    protocols.extract_and_evaluate_protocol('../config/renderman_tasks_for_ht2.py',
                                            '../config/l2_gabor_random_corr_test_comparison_model.py',
                                            '../config/ten_categories_images.py',
                                            convolve_func_name='numpy', write=True,parallel=True)
                                            


@protocolize()
def make_l2_gabor_corr_test_polygon_model(depends_on='../config/l2_gabor_corr_test_polygon_model.py'):
    """
    """
    protocols.model_protocol(depends_on,parallel=False,write=True)


@protocolize()
def ext_eval_l2_gabor_corr_test_polygon(depends_on=('../config/parallel_polygon_tasks_for_ht.py',
                                                  '../config/l2_gabor_corr_test_polygon_model.py',
                                                  '../config/polygon_task.py')):
    """
    bad results: 61%
    """
    protocols.extract_and_evaluate_protocol('../config/parallel_polygon_tasks_for_ht.py',
                                            '../config/l2_gabor_corr_test_polygon_model.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='numpy', write=True,parallel=True)


@protocolize()
def make_l2_gabor_corr_test_polygon_comparison_models(depends_on='../config/l2_gabor_corr_test_polygon_comparison_models.py'):
    """
    """
    protocols.model_protocol(depends_on,parallel=False,write=True)


@protocolize()
def ext_eval_l2_gabor_corr_test_polygon_comparison(depends_on=('../config/parallel_polygon_tasks_for_ht.py',
                                                  '../config/l2_gabor_corr_test_polygon_comparison_models.py',
                                                  '../config/polygon_task.py')):
    """
    shows that basically both the gabor/freq_uniform and gabor/random do significantly 
    better than the corr -- so corr method is wrong
    
    """
    protocols.extract_and_evaluate_protocol('../config/parallel_polygon_tasks_for_ht.py',
                                            '../config/l2_gabor_corr_test_polygon_comparison_models.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='numpy', write=True,parallel=True)


@protocolize()
def make_l2_gabor_corr_test_polygon_subset_model(depends_on='../config/l2_gabor_corr_test_polygon_subset_model.py'):
    """
    """
    protocols.model_protocol(depends_on,parallel=False,write=True)


@protocolize()
def ext_eval_l2_gabor_corr_test_polygon_subset(depends_on=('../config/parallel_polygon_tasks_for_ht.py',
                                                  '../config/l2_gabor_corr_test_polygon_subset_model.py',
                                                  '../config/polygon_task.py')):
    """
    now with random subsetting: 
    result: ok, back in the game, 81%
    """
    protocols.extract_and_evaluate_protocol('../config/parallel_polygon_tasks_for_ht.py',
                                            '../config/l2_gabor_corr_test_polygon_subset_model.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='numpy', write=True,parallel=True)


def make_l2_gabor_corr_test_polygon_subset_small_model(depends_on='../config/l2_gabor_corr_test_polygon_subset_model_small.py'):
    """
    """
    protocols.model_protocol(depends_on,parallel=False,write=True)


@protocolize()
def ext_eval_l2_gabor_corr_test_polygon_subset_small(depends_on=('../config/parallel_polygon_tasks_for_ht.py',
                                                  '../config/l2_gabor_corr_test_polygon_subset_model_small.py',
                                                  '../config/polygon_task.py')):
    """
    random subsetting with fewer l2 filters
    """
    protocols.extract_and_evaluate_protocol('../config/parallel_polygon_tasks_for_ht.py',
                                            '../config/l2_gabor_corr_test_polygon_subset_model_small.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='numpy', write=True,parallel=True)


@protocolize()
def make_l2_gabor_mult_test_polygon_model(depends_on='../config/l2_gabor_mult_test_polygon_model.py'):
    """
    """
    protocols.model_protocol(depends_on,parallel=False,write=True)


@protocolize()
def ext_eval_l2_gabor_mult_test_polygon(depends_on=('../config/parallel_polygon_tasks_for_test.py',
                                                  '../config/l2_gabor_mult_test_polygon_model.py',
                                                  '../config/polygon_task.py')):
    """
    trying it out with a kind of multiplation -- reults are bad
    """
    protocols.extract_and_evaluate_protocol('../config/parallel_polygon_tasks_for_test.py',
                                            '../config/l2_gabor_mult_test_polygon_model.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='numpy', write=True,parallel=True)

@protocolize()
def make_l2_gabor_mult_test_polygon_model2(depends_on='../config/l2_gabor_mult_test_polygon_model2.py'):
    """
    trying it out with a kind of multiplation -- reults are bad
    """
    protocols.model_protocol(depends_on,parallel=False,write=True)


@protocolize()
def ext_eval_l2_gabor_mult_test2_polygon(depends_on=('../config/parallel_polygon_tasks_for_test.py',
                                                  '../config/l2_gabor_mult_test_polygon_model2.py',
                                                  '../config/polygon_task.py')):
    """
    """
    protocols.extract_and_evaluate_protocol('../config/parallel_polygon_tasks_for_test.py',
                                            '../config/l2_gabor_mult_test_polygon_model2.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='numpy', write=True,parallel=True)


@protocolize()
def make_l2_gabor_mult_test_polygon_model_old(depends_on='../config/medium_gabor_mod4_with_multiplication_no_sum.py'):
    """
    """
    protocols.model_protocol(depends_on,parallel=False,write=True)


@protocolize()
def ext_eval_l2_gabor_mult_test_old_polygon(depends_on=('../config/parallel_polygon_tasks_for_test.py',
                                                  '../config/medium_gabor_mod4_with_multiplication_no_sum.py',
                                                  '../config/polygon_task.py')):
    """
    using old mult 
    """
    protocols.extract_and_evaluate_protocol('../config/parallel_polygon_tasks_for_test.py',
                                            '../config/medium_gabor_mod4_with_multiplication_no_sum.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='numpy', write=True,parallel=True)


@protocolize()
def make_l2_gabor_mult_test_polygon_model_old2(depends_on='../config/medium_gabor_then_multiplication.py'):
    """
    """
    protocols.model_protocol(depends_on,parallel=False,write=True)


@protocolize()
def ext_eval_l2_gabor_mult_test_old2_polygon(depends_on=('../config/parallel_polygon_tasks_for_test2.py',
                                                  '../config/medium_gabor_then_multiplication.py',
                                                  '../config/polygon_task.py')):
    """
    """
    protocols.extract_and_evaluate_protocol('../config/parallel_polygon_tasks_for_test2.py',
                                            '../config/medium_gabor_then_multiplication.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='numpy', write=True,parallel=True)
                                            
                                            
                                            
###########

@protocolize()
def ext_eval_l1_gabor_test_renderman(depends_on=('../config/renderman_tasks_for_l1_test.py',
                                                  '../config/l1_gabor_test_model.py',
                                                  '../config/ten_categories_images.py')):
    """
    the thing against which to compare corr test above - does equally well
    """
    protocols.extract_and_evaluate_protocol('../config/renderman_tasks_for_l1_test.py',
                                            '../config/l1_gabor_test_model.py',
                                            '../config/ten_categories_images.py',
                                            convolve_func_name='numpy', write=True,parallel=True)
                                                                                        
                                                                                        
###########
@protocolize()
def make_various_l2_gabor_gabor_models2(depends_on='../config/various_l2_gabor_gabor_models2.py'):
    """

    """
    protocols.model_protocol(depends_on,parallel=False,write=True) 
 
@protocolize()
def ext_eval_various_l2_gabor_gabor_2_renderman(depends_on=('../config/renderman_tasks_for_ht2.py',
                                                  '../config/various_l2_gabor_gabor_models2.py',
                                                  '../config/ten_categories_images.py')):
    """
    seeing how more 3d gabors do 
    nothing espec. great
    (37.600000000000001, 32.93333333333333, 34.977777777777781, 1.6450449354063783)
    """
    protocols.extract_and_evaluate_protocol('../config/renderman_tasks_for_ht2.py',
                                            '../config/various_l2_gabor_gabor_models2.py',
                                            '../config/ten_categories_images.py',
                                            convolve_func_name='numpy', write=True,parallel=True)
                                            
                                            
######

@protocolize()
def make_l2_gabor_eigen_test_polygon_model(depends_on='../config/l2_gabor_eigen_test_polygon_model.py'):
    """
    """
    protocols.model_protocol(depends_on,parallel=False,write=True)


@protocolize()
def ext_eval_l2_gabor_eigen_test_polygon(depends_on=('../config/parallel_polygon_tasks_for_eigentest.py',
                                                  '../config/l2_gabor_eigen_test_polygon_model.py',
                                                  '../config/polygon_task.py')):
    """
    seems to show that only a few filters is about as good as many here ... basically the nontrivial
    eigenvectors; but overall performance is not great.   (neither were many samples from the correlation
    distribution ....)
    """
    protocols.extract_and_evaluate_protocol('../config/parallel_polygon_tasks_for_eigentest.py',
                                            '../config/l2_gabor_eigen_test_polygon_model.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='numpy', write=True,parallel=True)

@protocolize()
def make_l2_gabor_eigen_test_polygon_anti_model(depends_on='../config/l2_gabor_eigen_test_polygon_anti_model.py'):
    """
    """
    protocols.model_protocol(depends_on,parallel=False,write=True)


@protocolize()
def ext_eval_l2_gabor_eigen_anti_test_polygon(depends_on=('../config/parallel_polygon_tasks_for_ht.py',
                                                  '../config/l2_gabor_eigen_test_polygon_anti_model.py',
                                                  '../config/polygon_task.py')):
    """
  
    """
    protocols.extract_and_evaluate_protocol('../config/parallel_polygon_tasks_for_ht.py',
                                            '../config/l2_gabor_eigen_test_polygon_anti_model.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='numpy', write=True,parallel=True)


@protocolize()
def make_l2_gabor_eigen_test_renderman_model(depends_on='../config/l2_gabor_eigen_test_renderman_model.py'):
    """
    """
    protocols.model_protocol(depends_on,parallel=False,write=True)


@protocolize()
def ext_eval_l2_gabor_eigen_test_renderman(depends_on=('../config/renderman_tasks_for_corr_test.py',
                                                  '../config/l2_gabor_eigen_test_renderman_model.py',
                                                  '../config/ten_categories_images.py')):
    """

    """
    protocols.extract_and_evaluate_protocol('../config/renderman_tasks_for_corr_test.py',
                                            '../config/l2_gabor_eigen_test_renderman_model.py',
                                            '../config/ten_categories_images.py',
                                            convolve_func_name='numpy', write=True,parallel=True)
                                            
####

@protocolize()
def make_l2_random_activation_models(depends_on='../config/l2_random_activation_models.py'):
    """
    """
    protocols.model_protocol(depends_on,parallel=False,write=True)


@protocolize()
def ext_eval_l2_random_activation_models_renderman(depends_on=('../config/renderman_tasks_for_corr_test.py',
                                                  '../config/l2_random_activation_models.py',
                                                  '../config/ten_categories_images.py')):
    """

    """
    protocols.extract_and_evaluate_protocol('../config/renderman_tasks_for_corr_test.py',
                                            '../config/l2_random_activation_models.py',
                                            '../config/ten_categories_images.py',
                                            convolve_func_name='numpy', write=True,parallel=True)



@protocolize()
def make_ht_l2_gabor_random_randomactivation_models(depends_on='../config/ht_l2_gabor_random_randomactivation_models.py'):
    """
  
    """
    protocols.model_protocol(depends_on,parallel=False,write=True)
 

@protocolize()
def ext_eval_ht_l2_gabor_random_randomactivation_renderman(depends_on=('../config/renderman_tasks_for_ht2.py',
                                                  '../config/ht_l2_gabor_random_randomactivation_models.py',
                                                  '../config/ten_categories_images.py')):
    """
    (47.333333333333329, 36.933333333333337, 42.983333333333327, 2.5525042169951)
     
    random activations cause a significant improvements, e.g comparison is:
    		(41.333333333333329, 38.000000000000007, 40.0, 1.225651754056678)
    """
    protocols.extract_and_evaluate_protocol('../config/renderman_tasks_for_ht2.py',
                                            '../config/ht_l2_gabor_random_randomactivation_models.py',
                                            '../config/ten_categories_images.py',
                                            convolve_func_name='numpy', write=True,parallel=True)

                                           
@protocolize()
def make_ht_l2_gabor_random_randomactivation_models2(depends_on='../config/ht_l2_gabor_random_randomactivation_models2.py'):
    """
  
    """
    protocols.model_protocol(depends_on,parallel=False,write=True)
 

@protocolize()
def ext_eval_ht_l2_gabor_random_randomactivation2_renderman(depends_on=('../config/renderman_tasks_for_ht2.py',
                                                  '../config/ht_l2_gabor_random_randomactivation_models2.py',
                                                  '../config/ten_categories_images.py')):
    """

    """
    protocols.extract_and_evaluate_protocol('../config/renderman_tasks_for_ht2.py',
                                            '../config/ht_l2_gabor_random_randomactivation_models2.py',
                                            '../config/ten_categories_images.py',
                                            convolve_func_name='numpy', write=True,parallel=True)

                                           
                                            
####                                            ####


@protocolize()
def ext_eval_various_random_l3_renderman_varied_subtasks(depends_on=('../config/renderman_subtasks_varied.py',
                                                  '../config/various_random_l3_grayscale_models.py',
                                                  '../config/ten_categories_images.py')):
    """

    """
    protocols.extract_and_evaluate_protocol('../config/renderman_subtasks_varied.py',
                                            '../config/various_random_l3_grayscale_models.py',
                                            '../config/ten_categories_images.py',
                                            convolve_func_name='numpy', write=True,parallel=True)
