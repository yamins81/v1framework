from starflow.protocols import Apply, Applies, protocolize

import deploy
import pythor_protocols as protocols
                                            
@protocolize()
def ext_eval_best_l3_exploratory_renderman(depends_on=('../config/exploratory_renderman_tasks.py',
                                                  '../config/best_l3_model.py',
                                                  '../config/ten_categories_images.py')):
    protocols.extract_and_evaluate_protocol('../config/exploratory_renderman_tasks.py',
                                            '../config/best_l3_model.py',
                                            '../config/ten_categories_images.py',
                                            convolve_func_name='numpy', write=True,parallel=False)
                                            


@protocolize()
def make_various_random_l3_models(depends_on='../config/various_random_l3_models.py'):
    protocols.model_protocol(depends_on,parallel=False,write=True)

@protocolize()
def ext_eval_various_random_l3_various_renderman(depends_on=('../config/parallel_renderman_tasks_for_random_l3.py',
                                                  '../config/various_random_l3_models.py',
                                                  '../config/ten_categories_images.py')):
    protocols.extract_and_evaluate_protocol('../config/parallel_renderman_tasks_for_random_l3.py',
                                            '../config/various_random_l3_models.py',
                                            '../config/ten_categories_images.py',
                                            convolve_func_name='numpy', write=True,parallel=True)
                                            
@protocolize()
def make_best_l3_grayscale_model(depends_on='../config/best_l3_grayscale_model.py'):
    """
    best l3 model but converts images to grayscale in preprocessing
    """
    protocols.model_protocol(depends_on,parallel=False,write=True)

@protocolize()
def ext_eval_best_l3_exploratory_grayscale_renderman(depends_on=('../config/exploratory_renderman_tasks2.py',
                                                  '../config/best_l3_grayscale_model.py',
                                                  '../config/ten_categories_images.py')):
    """
    testing grayscale best l3 on two hard renderman problems.  result:   about 29% for each problem
    """
    protocols.extract_and_evaluate_protocol('../config/exploratory_renderman_tasks2.py',
                                            '../config/best_l3_grayscale_model.py',
                                            '../config/ten_categories_images.py',
                                            convolve_func_name='numpy', write=True,parallel=False)





@protocolize()
def make_various_random_l3_grayscale_models(depends_on='../config/various_random_l3_grayscale_models.py'):
    """
    making a bunch (16) l3 random models with grayscale
    """
    protocols.model_protocol(depends_on,parallel=False,write=True)

@protocolize()
def ext_eval_various_random_l3_grayscale_various_polygon(depends_on=('../config/parallel_polygon_tasks_for_random_l3.py',
                                                  '../config/various_random_l3_grayscale_models.py',
                                                  '../config/polygon_task.py')):
    """
    testing a bunch of l3 random grayscale models on polygon problems:
    result: (a) all models clustered between 65-75% for rect vs. all problem (chance is 50%) 
            (b) all models clustered between 32-41% for 5-way choice problem (chacne is 20%)
    
    """
    protocols.extract_and_evaluate_protocol('../config/parallel_polygon_tasks_for_random_l3.py',
                                            '../config/various_random_l3_grayscale_models.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='numpy', write=True,parallel=True)
                                            
@protocolize()
def ext_eval_various_random_l3_grayscale_car_vs_plane(depends_on=('../config/exploratory_renderman_tasks_for_random_l3.py',
                                                  '../config/various_random_l3_grayscale_models.py',
                                                  '../config/ten_categories_images.py')):
    """
    
    """
    protocols.extract_and_evaluate_protocol('../config/exploratory_renderman_tasks_for_random_l3.py',
                                            '../config/various_random_l3_grayscale_models.py',
                                            '../config/ten_categories_images.py',
                                            convolve_func_name='numpy', write=True,parallel=True)

@protocolize()
def make_various_random_l2_grayscale_models(depends_on='../config/various_random_l2_grayscale_models.py'):
    """
    making a bunch of l2 random models with grayscale
    """
    protocols.model_protocol(depends_on,parallel=False,write=True)


@protocolize()
def ext_eval_various_random_l2_grayscale_various_polygon(depends_on=('../config/exploratory_polygon_tasks_for_random_l2.py',
                                                  '../config/various_random_l2_grayscale_models.py',
                                                  '../config/polygon_task.py')):
    """

    
    """
    protocols.extract_and_evaluate_protocol('../config/exploratory_polygon_tasks_for_random_l2.py',
                                            '../config/various_random_l2_grayscale_models.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='numpy', write=True,parallel=True)
                                            
def ext_eval_various_random_l2_grayscale_various_renderman(depends_on=('../config/exploratory_renderman_tasks_for_random_l2.py',
                                                  '../config/various_random_l2_grayscale_models.py',
                                                  '../config/ten_categories_images.py')):
    """
    
    """
    protocols.extract_and_evaluate_protocol('../config/exploratory_renderman_tasks_for_random_l2.py',
                                            '../config/various_random_l2_grayscale_models.py',
                                            '../config/ten_categories_images.py',
                                            convolve_func_name='numpy', write=True,parallel=True)

                                            
@protocolize()
def make_small_gabor_then_random_model(depends_on='../config/small_gabor_then_random_model.py'):
    """
    small gabor then random model
    """
    protocols.model_protocol(depends_on,parallel=False,write=True)

@protocolize()
def ext_eval_small_gabor_then_random_exploratory_renderman(depends_on=('../config/exploratory_renderman_tasks2.py',
                                                  '../config/small_gabor_then_random_model.py',
                                                  '../config/ten_categories_images.py')):
    """
    testing small gabor then random on two hard renderman problems.  
    result:   
    """
    protocols.extract_and_evaluate_protocol('../config/exploratory_renderman_tasks2.py',
                                            '../config/small_gabor_then_random_model.py',
                                            '../config/ten_categories_images.py',
                                            convolve_func_name='numpy', write=True,parallel=False)

@protocolize()
def ext_eval_small_gabor_then_random_exploratory_polygon(depends_on=('../config/exploratory_polygon_tasks.py',
                                                  '../config/small_gabor_then_random_model.py',
                                                  '../config/polygon_task.py')):
    """
    testing small gabor then random on two hard renderman problems.  
    result:   
    """
    protocols.extract_and_evaluate_protocol('../config/exploratory_polygon_tasks.py',
                                            '../config/small_gabor_then_random_model.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='numpy', write=True,parallel=False)
                                            

@protocolize()
def make_various_random_l3_grayscale_models2(depends_on='../config/various_random_l3_grayscale_models2.py'):
    """
    making a bunch more l3 random models with grayscale
    """
    protocols.model_protocol(depends_on,parallel=False,write=True)


@protocolize()
def ext_eval_various_random_l3_grayscale2_various_polygon(depends_on=('../config/parallel_polygon_tasks_for_random_l3.py',
                                                  '../config/various_random_l3_grayscale_models2.py',
                                                  '../config/polygon_task.py')):
    """
    testing a bunch more l3 random grayscale models on polygon problems:
    results: 
    	-- max/min results are about 75%/50% for the rect vs everythign problem and 45%/22% for the five-way choice
    	-- 9 is much better than 3 for l3 pool kershape for both problems
    	-- 5,7 predominate over 3 in l2 filter kershape for the two-choice problem but not for the five-choice
    	-- 3,5 predominate over 7 in l3 filter kershape for the five-choice problem but not so much for the two-shoice
    	-- l2 norm stretch doesn't matter much for either problem 
    	-- l2 norm shape doesn't matter much for either problem
    
    """
    protocols.extract_and_evaluate_protocol('../config/parallel_polygon_tasks_for_random_l3.py',
                                            '../config/various_random_l3_grayscale_models2.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='numpy', write=True,parallel=True)
                                            
                                            

@protocolize()
def make_various_l2_freq_models(depends_on='../config/various_l2_freq_models.py'):
    """
  
    """
    protocols.model_protocol(depends_on,parallel=False,write=True)
    
@protocolize()
def ext_eval_various_l2_freq_exploratory_renderman(depends_on=('../config/exploratory_renderman_tasks_for_freq.py',
                                                  '../config/various_l2_freq_models.py',
                                                  '../config/ten_categories_images.py')):
    """

    
    """
    protocols.extract_and_evaluate_protocol('../config/exploratory_renderman_tasks_for_freq.py',
                                            '../config/various_l2_freq_models.py',
                                            '../config/ten_categories_images.py',
                                            convolve_func_name='numpy', write=True,parallel=True)

@protocolize()
def make_various_l2_freq_models2(depends_on='../config/various_l2_freq_models2.py'):
    """

    """
    protocols.model_protocol(depends_on,parallel=False,write=True)
    
@protocolize()
def ext_eval_various_l2_freq_exploratory_renderman2(depends_on=('../config/exploratory_renderman_tasks_for_freq.py',
                                                  '../config/various_l2_freq_models2.py',
                                                  '../config/ten_categories_images.py')):
    """

    
    """
    protocols.extract_and_evaluate_protocol('../config/exploratory_renderman_tasks_for_freq.py',
                                            '../config/various_l2_freq_models2.py',
                                            '../config/ten_categories_images.py',
                                            convolve_func_name='numpy', write=True,parallel=True)

@protocolize()
def ext_eval_various_l2_freq_harder_renderman(depends_on=('../config/exploratory_renderman_tasks_for_freq2.py',
                                                  '../config/various_l2_freq_models.py',
                                                  '../config/ten_categories_images.py')):
    """

    
    """
    protocols.extract_and_evaluate_protocol('../config/exploratory_renderman_tasks_for_freq2.py',
                                            '../config/various_l2_freq_models.py',
                                            '../config/ten_categories_images.py',
                                            convolve_func_name='numpy', write=True,parallel=True)

@protocolize()
def ext_eval_various_l2_freq_various_polygon(depends_on=('../config/parallel_polygon_tasks_for_freq.py',
                                                  '../config/various_l2_freq_models.py',
                                                  '../config/polygon_task.py')):
    """

    
    """
    protocols.extract_and_evaluate_protocol('../config/parallel_polygon_tasks_for_freq.py',
                                            '../config/various_l2_freq_models.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='numpy', write=True,parallel=True)