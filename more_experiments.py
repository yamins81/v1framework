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
    testing grayscale best l3 on two hard renderman problems.
    results:
    	faces (chance is 12.5%): 29.4%
    	10-way (chance is 10%): 29%
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
    results:
    	5-way choice (chance is 20%):
    		max: 41.6%   -- up go 54% with max transform avg.
    		min: 27.6% 
    		mean: 36%
    		std: 3.5%
		rect vs. all (chance is 50%):
			max: 75.8% 
			min: 62.5%
			mean: 68%
			std: 3.79%
		rect vs. all with trans with max(chance is 50%):
			max: 86.25
			min: 57.5
			mean: 73
			std: 6.38
    
    """
    protocols.extract_and_evaluate_protocol('../config/parallel_polygon_tasks_for_random_l3.py',
                                            '../config/various_random_l3_grayscale_models.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='numpy', write=True,parallel=True)
                                            
@protocolize()
def ext_eval_various_random_l3_renderman_subtasks(depends_on=('../config/renderman_subtasks_for_freq.py',
                                                  '../config/various_random_l3_grayscale_models.py',
                                                  '../config/ten_categories_images.py')):
    """

    
    """
    protocols.extract_and_evaluate_protocol('../config/renderman_subtasks_for_freq.py',
                                            '../config/various_random_l3_grayscale_models.py',
                                            '../config/ten_categories_images.py',
                                            convolve_func_name='numpy', write=True,parallel=True)

                                            
@protocolize()
def ext_eval_various_random_l3_grayscale_car_vs_plane(depends_on=('../config/exploratory_renderman_tasks_for_random_l3.py',
                                                  '../config/various_random_l3_grayscale_models.py',
                                                  '../config/ten_categories_images.py')):
    """
    hash 87736f529ff7c9b1df90f8be4b3f25dc31383fa5
    result: max: 78.75%
    		min: 57.5%
    		mean: 67.5%
    		std: 5.6%
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
    5-way:
        max: 37
        min: 12.0
        mean: 23.6
        std: 4.7
    5-way with trans avg max:
        max: 56.5
        min: 36.5
        mean: 46.5
        std: 4.4
    Rect vs all:
        max: 70.0
        min: 40
        mean: 56.9
        std: 6.2
    Rect vs all with trans. avg max:
        max: 88.75
        min: 62.5
        mean: 73.1
        Std: 5.5
    

    
    """
    protocols.extract_and_evaluate_protocol('../config/exploratory_polygon_tasks_for_random_l2.py',
                                            '../config/various_random_l2_grayscale_models.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='numpy', write=True,parallel=True)

@protocolize()                                            
def ext_eval_various_random_l2_grayscale_various_renderman(depends_on=('../config/exploratory_renderman_tasks_for_random_l2.py',
                                                  '../config/various_random_l2_grayscale_models.py',
                                                  '../config/ten_categories_images.py')):
    """
    faces:
        max: 23.1
        min: 12.5
        mean: 18.73
        std: 2.5
    10-way:
        max: 33.5
        min: 14.75
        mean: 23.69
        std: 3.97
    Car vs plane:
        max: 80
        min: 51
        mean: 65.8
        std: 5.9
    Car vs plane with trans avg, max:
        max: 73.75
        min: 42.6
        mean: 62.7
        std: 6.1
    Car vs plane with trans avg:
        max: 75
        min: 46.26
        mean: 64
        std: 5.2
        
    
    """
    protocols.extract_and_evaluate_protocol('../config/exploratory_renderman_tasks_for_random_l2.py',
                                            '../config/various_random_l2_grayscale_models.py',
                                            '../config/ten_categories_images.py',
                                            convolve_func_name='numpy', write=True,parallel=True)

@protocolize()
def ext_eval_various_random_l2_renderman_subtasks(depends_on=('../config/renderman_subtasks_for_freq.py',
                                                  '../config/various_random_l2_grayscale_models.py',
                                                  '../config/ten_categories_images.py')):
    """

    
    """
    protocols.extract_and_evaluate_protocol('../config/renderman_subtasks_for_freq.py',
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
    cars vs planes, trans with max:
    	max: 73.75
    	min: 47.5
        mean: 62.5
        std: 7.3
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
    cars vs planes, trans with max:
    	max: 70
    	min: 51.25
    	mean: 58.5
    	std: 4.81
    
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
    faces (12.5% chance):
        max: 26.56 
        min: 18.75
        mean: 22.79
        std: 2.43
    10-way:
    	max: 24
    	min: 15.75
    	mean: 19.9
    	std: 2.25

    
    """
    protocols.extract_and_evaluate_protocol('../config/exploratory_renderman_tasks_for_freq2.py',
                                            '../config/various_l2_freq_models.py',
                                            '../config/ten_categories_images.py',
                                            convolve_func_name='numpy', write=True,parallel=True)

@protocolize()
def ext_eval_various_l2_freq_renderman_subtasks(depends_on=('../config/renderman_subtasks_for_freq.py',
                                                  '../config/various_l2_freq_models.py',
                                                  '../config/ten_categories_images.py')):
    """

    
    """
    protocols.extract_and_evaluate_protocol('../config/renderman_subtasks_for_freq.py',
                                            '../config/various_l2_freq_models.py',
                                            '../config/ten_categories_images.py',
                                            convolve_func_name='numpy', write=True,parallel=True)


@protocolize()
def ext_eval_various_l2_freq_various_polygon(depends_on=('../config/parallel_polygon_tasks_for_freq.py',
                                                  '../config/various_l2_freq_models.py',
                                                  '../config/polygon_task.py')):
    """
    5-way choice (chance is 20%):
        max: 89.5%
        min: 55.5%
        mean: 76.6%
        std: 9.9%
    rect vs all (chance is 50%):
        max: 92.5%
        min: 67.5%
        mean: 84%
        std: 6.4%
    
    """
    protocols.extract_and_evaluate_protocol('../config/parallel_polygon_tasks_for_freq.py',
                                            '../config/various_l2_freq_models.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='numpy', write=True,parallel=True)
                                            
                                            

@protocolize()
def make_medium_gabor_mod4_sameconv_model(depends_on='../config/medium_gabor_mod4_with_freq_uniform_sameconv.py'):
    """

    """
    protocols.model_protocol(depends_on,parallel=False,write=True)



@protocolize()
def ext_eval_medium_gabor_mod4_sameconv_freq_uniform_ten_categories(depends_on=('../config/cars_vs_planes_task_for_freq.py',
                                                  '../config/medium_gabor_mod4_with_freq_uniform_sameconv.py',
                                                  '../config/ten_categories_images.py')):
    protocols.extract_and_evaluate_protocol('../config/cars_vs_planes_task_for_freq.py',
                                            '../config/medium_gabor_mod4_with_freq_uniform_sameconv.py',
                                            '../config/ten_categories_images.py',
                                            convolve_func_name='numpy', write=True,parallel=False)
                                            
@protocolize()
def make_small_gabor_sameconv_model(depends_on='../config/small_gabor_with_freq_uniform_sameconv.py'):
    """

    """
    protocols.model_protocol(depends_on,parallel=False,write=True)



@protocolize()
def ext_eval_small_gabor_sameconv_freq_uniform_ten_categories(depends_on=('../config/cars_vs_planes_task_for_freq2.py',
                                                  '../config/small_gabor_with_freq_uniform_sameconv.py',
                                                  '../config/ten_categories_images.py')):
    protocols.extract_and_evaluate_protocol('../config/cars_vs_planes_task_for_freq2.py',
                                            '../config/small_gabor_with_freq_uniform_sameconv.py',
                                            '../config/ten_categories_images.py',
                                            convolve_func_name='numpy', write=True,parallel=False)
