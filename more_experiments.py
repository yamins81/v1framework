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
    		max: 41.6%   -- up to 54% with max transform avg.
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
        no out of plane rot:
            max,min,mean,std : 85.0, 67.5, 76.09375, 4.6837369154400159
        not rot: 
            max,min,mean,std : 85.0, 73.333333333333329, 80.78125, 3.7205877811845829
        limited trans:
            max,min,mean,std : 80.833333333333329, 66.666666666666671, 72.447916666666657, 3.9798619593244
        limited scale: 
            max,min,mean,std : 78.333333333333329, 64.166666666666671, 71.510416666666657, 4.1793426973561818
    
    """
    protocols.extract_and_evaluate_protocol('../config/renderman_subtasks_for_freq.py',
                                            '../config/various_random_l3_grayscale_models.py',
                                            '../config/ten_categories_images.py',
                                            convolve_func_name='numpy', write=True,parallel=True)


@protocolize()
def ext_eval_various_random_l3_renderman_subtasks_more(depends_on=('../config/renderman_subtasks.py',
                                                  '../config/various_random_l3_grayscale_models.py',
                                                  '../config/ten_categories_images.py')):
    """
    10-way
    just translation:
        (85.666666666666671, 75.666666666666671, 81.791666666666671, 2.6070817018267758)
    just in-plane rotation:
        (53.333333333333336, 41.0, 47.458333333333329, 3.1773394705493958)
    just out of plane rotation:
    	(52.0, 40.0, 45.802083333333336, 3.6501611218243197)
    just scaling
    	(81.0, 67.833333333333329, 76.520833333333329, 3.728584684568665)
    scaling and in plane rotation
    	(41.833333333333336, 32.5, 36.78125, 2.5838163190972643)
    translation and inplane rotation
    	(47.333333333333336, 37.333333333333336, 41.583333333333343, 2.7726341266023535)
    all rotation:
    	(35.833333333333336, 29.5, 32.750000000000007, 1.764818215378948)
    """
    protocols.extract_and_evaluate_protocol('../config/renderman_subtasks.py',
                                            '../config/various_random_l3_grayscale_models.py',
                                            '../config/ten_categories_images.py',
                                            convolve_func_name='numpy', write=True,parallel=True)

@protocolize()
def ext_eval_various_random_l3_polygon_subtasks(depends_on=('../config/polygon_subtasks.py',
                                                  '../config/various_random_l3_grayscale_models.py',
                                                  '../config/polygon_task.py')):
    """
    5way
    just translation:
    	98.7,88.3,94.5,3.11
    just rotation:
    	86.3,69,75,4.7
    just scale:
    	99.7,90,96.2,2.43
    rotation and scale:
        71.6,53,62.6,5.98
    translation and rotation:
        62.3,45.0,55.2,4.7
    	
    	
    rect:
    just translation:
    	100.0,91.6,96.4,2.1
    just rotation:
        92.5,76.7,83.9,5.3
    just scale:
    	100,95,98,1.2
    rotation and scale:
    	89.1,70,80,5.15
    translation and rotation:
    	86.6,69.1,78.2,5.2
    	
    """
    
    protocols.extract_and_evaluate_protocol('../config/polygon_subtasks.py',
                                            '../config/various_random_l3_grayscale_models.py',
                                            '../config/polygon_task.py',
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
def make_various_l2_gabor_random_models(depends_on='../config/various_l2_gabor_random_models.py'):
    """
  
    """
    protocols.model_protocol(depends_on,parallel=False,write=True)
    
@protocolize()
def make_various_l2_freq_from_l1_gabor_models(depends_on='../config/various_l2_freq_from_l1_gabor_models.py'):
    """
  
    """
    protocols.model_protocol(depends_on,parallel=False,write=True)

@protocolize()
def make_various_l2_random_random_models(depends_on='../config/various_l2_random_random_models.py'):
    """
  
    """
    protocols.model_protocol(depends_on,parallel=False,write=True)

@protocolize()
def make_more_l2_freq_models(depends_on='../config/more_l2_freq_models.py'):
    """
  
    """
    protocols.model_protocol(depends_on,parallel=False,write=True)


@protocolize()
def make_more_l2_gabor_random_models(depends_on='../config/more_l2_gabor_random_models.py'):
    """
  
    """
    protocols.model_protocol(depends_on,parallel=False,write=True)



@protocolize()
def make_various_big_l2_random_random_models(depends_on='../config/various_big_l2_random_random_models.py'):
    """
  
    """
    protocols.model_protocol(depends_on,parallel=False,write=True)
    

@protocolize()
def make_ht_l2_freq_models(depends_on='../config/ht_l2_freq_models.py'):
    """
  
    """
    protocols.model_protocol(depends_on,parallel=False,write=True)
    
@protocolize()
def make_ht_l2_random_random_models(depends_on='../config/ht_l2_random_random_models.py'):
    """
  
    """
    protocols.model_protocol(depends_on,parallel=False,write=True)
   
@protocolize()
def make_ht_l2_freq_models_2(depends_on='../config/ht_l2_freq_models_2.py'):
    """
  
    """
    protocols.model_protocol(depends_on,parallel=False,write=True)

@protocolize()
def make_ht_l1_gabor_models(depends_on='../config/ht_l1_gabor_models.py'):
    """
  
    """
    protocols.model_protocol(depends_on,parallel=False,write=True)


@protocolize()
def make_ht_l2_gabor_random_models(depends_on='../config/ht_l2_gabor_random_models.py'):
    """
  
    """
    protocols.model_protocol(depends_on,parallel=False,write=True)

@protocolize()
def make_ht_l1_random_models(depends_on='../config/ht_l1_random_models.py'):
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
    no out of plane rot:
    	max,min,mean,std : 80.0, 68.333333333333329, 73.333333333333329, 3.4291776360024366
    not rot at all:
        max,min,mean,std : 85.833333333333329, 72.5, 79.111111111111114, 3.7449554547450479
    limited translation:
        max,min,mean,std : 74.166666666666671, 59.166666666666664, 68.555555555555557 4.0764302950764773
    limited scale:
        max,min,mean,std : 80.833333333333329, 61.666666666666664, 69.277777777777771, .0080182621053053
    """
    protocols.extract_and_evaluate_protocol('../config/renderman_subtasks_for_freq.py',
                                            '../config/various_l2_freq_models.py',
                                            '../config/ten_categories_images.py',
                                            convolve_func_name='numpy', write=True,parallel=True)

@protocolize()
def ext_eval_various_l2_freq_renderman_subtasks_more(depends_on=('../config/renderman_subtasks.py',
                                                  '../config/various_l2_freq_models.py',
                                                  '../config/ten_categories_images.py')):
    """
    10-way
    just translation:
        85.7,67,80.2,4.65
    just in-plane rotation:
        46.166666666666664, 36.5, 41.24444444444444, 2.7331074884112909
    just out of plane rotation:
    	42.0, 28.5, 35.211111111111109, 3.0577567828736596
    just scaling
    	78.5, 63.333333333333336, 73.077777777777769, 4.5529260703935366
    scaling and in plane rotation
    	(34.166666666666664, 24.5, 30.988888888888887, 2.7436571070357272)
    translation and inplane rotation
    	(39.666666666666664, 30.833333333333332, 34.700000000000003, 2.6170381901854025)
    all rotation:
    	(28.5, 21.5, 25.099999999999998, 1.997034838992086)
    
    """
    protocols.extract_and_evaluate_protocol('../config/renderman_subtasks.py',
                                            '../config/various_l2_freq_models.py',
                                            '../config/ten_categories_images.py',
                                            convolve_func_name='numpy', write=True,parallel=True)

@protocolize()
def ext_eval_various_l2_freq_polygon_subtasks(depends_on=('../config/polygon_subtasks.py',
                                                  '../config/various_l2_freq_models.py',
                                                  '../config/polygon_task.py')):
    """
    5way
    just translation:
    	100,97.6,99.2,.728
    just rotation:
    	97,71,91,7.1
    just scale:
    	100,95.0,98.2,1.23
    rotation and scale:
        93,55,82,8.7
    translation and rotation:
    	99,73,92,6.8
    	
    rect:
    just translation:
    	100,92.5,97.6,2.15
    just rotation:
        98.3,74.1,90.5,5.52
    just scale:
    	99.1,92.5,96.5,1.89
    rotation and scale:
    	92.5,70.8,86.2,5.2
    translation and rotation:
    	95.8,70.0,88.8,6.9
    	
    	
    """
    protocols.extract_and_evaluate_protocol('../config/polygon_subtasks.py',
                                            '../config/various_l2_freq_models.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='numpy', write=True,parallel=True)


@protocolize()
def ext_eval_various_l2_freq_various_polygon(depends_on=('../config/parallel_polygon_tasks_for_freq.py',
                                                  '../config/various_l2_freq_models.py',
                                                  '../config/polygon_task.py')):
    """
    5-way choice (chance is 20%):
        89.5%, 55.5%, 76.6%, 9.9%
    rect vs all (chance is 50%):
        92.5%, 67.5%, 84%, 6.4%
    
    """
    protocols.extract_and_evaluate_protocol('../config/parallel_polygon_tasks_for_freq.py',
                                            '../config/various_l2_freq_models.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='numpy', write=True,parallel=True)


@protocolize()
def ext_eval_various_l2_gabor_random_various_polygon(depends_on=('../config/parallel_polygon_tasks_for_freq.py',
                                                  '../config/various_l2_gabor_random_models.py',
                                                  '../config/polygon_task.py')):
    """
    5way:
    	(87.5, 47.0, 74.321428571428569, 10.709820498124177)
    Rect:
    	(90.0, 65.0, 82.232142857142861, 5.6334969836954842)
    
    """
    protocols.extract_and_evaluate_protocol('../config/parallel_polygon_tasks_for_freq.py',
                                            '../config/various_l2_gabor_random_models.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='numpy', write=True,parallel=True)

@protocolize()
def ext_eval_various_l2_random_random_various_polygon(depends_on=('../config/parallel_polygon_tasks_for_freq.py',
                                                  '../config/various_l2_random_random_models.py',
                                                  '../config/polygon_task.py')):
    """
    5 way
        (76.5, 46.0, 66.15384615384616, 7.9551256224297964)
    Rect:
    	(88.75, 70.0, 80.09615384615384, 4.9386023224600999)
        
    """
    protocols.extract_and_evaluate_protocol('../config/parallel_polygon_tasks_for_freq.py',
                                            '../config/various_l2_random_random_models.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='numpy', write=True,parallel=True)                                            

@protocolize()
def ext_eval_various_l2_freq_from_l1_gabor_various_polygon(depends_on=('../config/parallel_polygon_tasks_for_freq.py',
                                                  '../config/various_l2_freq_from_l1_gabor_models.py',
                                                  '../config/polygon_task.py')):
    """
    5way 
        (89.5, 53.0, 75.961538461538467, 11.016528733019481)
    Rect:
    	(93.75, 75.0, 82.692307692307693, 6.1598528377103721)
    
    """
    protocols.extract_and_evaluate_protocol('../config/parallel_polygon_tasks_for_freq.py',
                                            '../config/various_l2_freq_from_l1_gabor_models.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='numpy', write=True,parallel=True)

@protocolize()
def ext_eval_various_l2_freq_from_l1_gabor_various_polygon_more_reps(depends_on=('../config/parallel_polygon_tasks_for_freq_morereps2.py',
                                                  '../config/various_l2_freq_from_l1_gabor_models.py',
                                                  '../config/polygon_task.py')):
    """
    5way:
        (93.200000000000003, 66.5, 82.58461538461539, 8.8156195431174371)
    rect
    	(94.75, 80.25, 89.134615384615387, 4.3007327021273865)
    """
    protocols.extract_and_evaluate_protocol('../config/parallel_polygon_tasks_for_freq_morereps2.py',
                                            '../config/various_l2_freq_from_l1_gabor_models.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='numpy', write=True,parallel=True)

@protocolize()
def ext_eval_various_big_l2_random_random_various_polygon(depends_on=('../config/parallel_polygon_tasks_for_freq.py',
                                                  '../config/various_big_l2_random_random_models.py',
                                                  '../config/polygon_task.py')):
    """
    5way 
        (77.0, 41.5, 65.433333333333337, 11.607277410697517)
    Rect:
    	(86.25, 66.25, 77.833333333333329, 5.0717080182343128)

    """
    protocols.extract_and_evaluate_protocol('../config/parallel_polygon_tasks_for_freq.py',
                                            '../config/various_big_l2_random_random_models.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='numpy', write=True,parallel=True) 
                                            
@protocolize()
def ext_eval_more_l2_freq_various_polygon(depends_on=('../config/parallel_polygon_tasks_for_freq_morereps.py',
                                                  '../config/more_l2_freq_models.py',
                                                  '../config/polygon_task.py')):
    """
    5way:
    (85.75, 61.25, 76.277777777777771, 8.4626995663687179)
    
    rect:
    (92.5, 73.75, 85.902777777777771, 5.4733501375661193)
    
    """
    protocols.extract_and_evaluate_protocol('../config/parallel_polygon_tasks_for_freq_morereps.py',
                                            '../config/more_l2_freq_models.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='numpy', write=True,parallel=True)

@protocolize()
def ext_eval_more_l2_gabor_random_various_polygon(depends_on=('../config/parallel_polygon_tasks_for_freq_morereps.py',
                                                  '../config/more_l2_gabor_random_models.py',
                                                  '../config/polygon_task.py')):
    """
    5way   
    (86.0, 63.0, 78.083333333333329, 6.7144123594945615)
    Rect
    (91.875, 77.5, 84.930555555555557, 4.1083887382636224)
    
    """
    protocols.extract_and_evaluate_protocol('../config/parallel_polygon_tasks_for_freq_morereps.py',
                                            '../config/more_l2_gabor_random_models.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='numpy', write=True,parallel=True)


@protocolize()
def ext_eval_ht_l2_freq_polygon(depends_on=('../config/parallel_polygon_tasks_for_ht.py',
                                                  '../config/ht_l2_freq_models.py',
                                                  '../config/polygon_task.py')):
    """

    
    """
    protocols.extract_and_evaluate_protocol('../config/parallel_polygon_tasks_for_ht.py',
                                            '../config/ht_l2_freq_models.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='numpy', write=True,parallel=True)

@protocolize()
def ext_eval_ht_l2_random_random_polygon(depends_on=('../config/parallel_polygon_tasks_for_ht.py',
                                                  '../config/ht_l2_random_random_models.py',
                                                  '../config/polygon_task.py')):
    """

    
    """
    protocols.extract_and_evaluate_protocol('../config/parallel_polygon_tasks_for_ht.py',
                                            '../config/ht_l2_random_random_models.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='numpy', write=True,parallel=True)

@protocolize()
def ext_eval_ht_l2_freq_2_polygon(depends_on=('../config/parallel_polygon_tasks_for_ht.py',
                                                  '../config/ht_l2_freq_models_2.py',
                                                  '../config/polygon_task.py')):
    """

    
    """
    protocols.extract_and_evaluate_protocol('../config/parallel_polygon_tasks_for_ht.py',
                                            '../config/ht_l2_freq_models_2.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='numpy', write=True,parallel=True)

@protocolize()
def ext_eval_ht_l1_random_polygon(depends_on=('../config/parallel_polygon_tasks_for_ht.py',
                                                  '../config/ht_l1_random_models.py',
                                                  '../config/polygon_task.py')):
    """

    
    """
    protocols.extract_and_evaluate_protocol('../config/parallel_polygon_tasks_for_ht.py',
                                            '../config/ht_l1_random_models.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='numpy', write=True,parallel=True)

@protocolize()
def ext_eval_ht_l2_gabor_random_polygon(depends_on=('../config/parallel_polygon_tasks_for_ht.py',
                                                  '../config/ht_l2_gabor_random_models.py',
                                                  '../config/polygon_task.py')):
    """

    
    """
    protocols.extract_and_evaluate_protocol('../config/parallel_polygon_tasks_for_ht.py',
                                            '../config/ht_l2_gabor_random_models.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='numpy', write=True,parallel=True)

@protocolize()
def ext_eval_ht_l1_gabor_polygon(depends_on=('../config/parallel_polygon_tasks_for_ht.py',
                                                  '../config/ht_l1_gabor_models.py',
                                                  '../config/polygon_task.py')):
    """

    
    """
    protocols.extract_and_evaluate_protocol('../config/parallel_polygon_tasks_for_ht.py',
                                            '../config/ht_l1_gabor_models.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='numpy', write=True,parallel=True)
                                            

@protocolize()
def make_medium_gabor_mod4_sameconv_model(depends_on='../config/medium_gabor_mod4_with_freq_uniform_sameconv.py'):

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





@protocolize()
def make_l2_freq_test_model(depends_on='../config/l2_freq_test_model.py'):
    """
    
    """
    protocols.model_protocol(depends_on,parallel=False,write=True)
    

@protocolize()
def ext_eval_l2_freq_test_polygon(depends_on=('../config/parallel_polygon_tasks_for_ht.py',
                                                  '../config/l2_freq_test_model.py',
                                                  '../config/polygon_task.py')):
    """
    confirming performance of a good l2_freq with 10 trials; 86.5% (confirmed)  	
    
    """
    protocols.extract_and_evaluate_protocol('../config/parallel_polygon_tasks_for_ht.py',
                                            '../config/l2_freq_test_model.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='numpy', write=True,parallel=False)


@protocolize()
def make_l2_freq_test_model_2(depends_on='../config/l2_freq_test_model_2.py'):
    """
    

    """
    protocols.model_protocol(depends_on,parallel=False,write=True)
    

@protocolize()
def ext_eval_l2_freq_test_2_polygon(depends_on=('../config/parallel_polygon_tasks_for_ht.py',
                                                  '../config/l2_freq_test_model_2.py',
                                                  '../config/polygon_task.py')):
    """
    showing that changing l1 pool order from 2 to 1 drops performance by more than 10% to 75.3%
    
    """
    protocols.extract_and_evaluate_protocol('../config/parallel_polygon_tasks_for_ht.py',
                                            '../config/l2_freq_test_model_2.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='numpy', write=True,parallel=True)


@protocolize()
def make_l2_freq_test_model_3(depends_on='../config/l2_freq_test_model_3.py'):
    """
    
    """
    protocols.model_protocol(depends_on,parallel=False,write=True)
    

@protocolize()
def ext_eval_l2_freq_test_3_polygon(depends_on=('../config/parallel_polygon_tasks_for_ht.py',
                                                  '../config/l2_freq_test_model_3.py',
                                                  '../config/polygon_task.py')):
    """
    taking best l2 freq from highthrouput run and changing l1 pool order, performance climbs almost 10% to 88.8%
    
    """
    protocols.extract_and_evaluate_protocol('../config/parallel_polygon_tasks_for_ht.py',
                                            '../config/l2_freq_test_model_3.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='numpy', write=True,parallel=True)
                                            

@protocolize()
def make_l2_freq_test_model_4(depends_on='../config/l2_freq_test_model_4.py'):
    """

    """
    protocols.model_protocol(depends_on,parallel=False,write=True)
    

@protocolize()
def ext_eval_l2_freq_test_4_polygon(depends_on=('../config/parallel_polygon_tasks_for_ht.py',
                                                  '../config/l2_freq_test_model_4.py',
                                                  '../config/polygon_task.py')):
    """
    seeing if changing l1pool order even higher has big effect (nope, perf is 87.9%)
    
    """
    protocols.extract_and_evaluate_protocol('../config/parallel_polygon_tasks_for_ht.py',
                                            '../config/l2_freq_test_model_4.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='numpy', write=True,parallel=True)

@protocolize()
def make_l2_gabor_random_test_model(depends_on='../config/l2_gabor_random_test_model.py'):
    """

    """
    protocols.model_protocol(depends_on,parallel=False,write=True)
    

@protocolize()
def ext_eval_l2_gabor_random_test_polygon(depends_on=('../config/parallel_polygon_tasks_for_ht.py',
                                                  '../config/l2_gabor_random_test_model.py',
                                                  '../config/polygon_task.py')):
    """
    seeing if changing l1pool order influences ersult of gabor random model
    
    """
    protocols.extract_and_evaluate_protocol('../config/parallel_polygon_tasks_for_ht.py',
                                            '../config/l2_gabor_random_test_model.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='numpy', write=True,parallel=True)
                                            


@protocolize()
def make_l2_random_random_test_model(depends_on='../config/l2_random_random_test_model.py'):
    """

    """
    protocols.model_protocol(depends_on,parallel=False,write=True)
    

@protocolize()
def ext_eval_l2_random_random_test_polygon(depends_on=('../config/parallel_polygon_tasks_for_ht.py',
                                                  '../config/l2_random_random_test_model.py',
                                                  '../config/polygon_task.py')):
    """
    seeing of changing l1pool order to 2 drops performance (yes, by 5% to 79%)
    
    """
    protocols.extract_and_evaluate_protocol('../config/parallel_polygon_tasks_for_ht.py',
                                            '../config/l2_random_random_test_model.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='numpy', write=True,parallel=True)
                                            
@protocolize()
def make_l2_random_random_test_model_2(depends_on='../config/l2_random_random_test_model_2.py'):
    """

    """
    protocols.model_protocol(depends_on,parallel=False,write=True)
    

@protocolize()
def ext_eval_l2_random_random_test_2_polygon(depends_on=('../config/parallel_polygon_tasks_for_ht.py',
                                                  '../config/l2_random_random_test_model_2.py',
                                                  '../config/polygon_task.py')):
    """
    seeing if increasing number of filters (doubled in each case) increases oerformance of best random l2
    answer; no, says at 84%
    
    """
    protocols.extract_and_evaluate_protocol('../config/parallel_polygon_tasks_for_ht.py',
                                            '../config/l2_random_random_test_model_2.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='numpy', write=True,parallel=True) 
                                            
                                            
@protocolize()
def make_l2_random_random_test_model_3(depends_on='../config/l2_random_random_test_model_3.py'):
    """

    """
    protocols.model_protocol(depends_on,parallel=False,write=True)
    

@protocolize()
def ext_eval_l2_random_random_test_3_polygon(depends_on=('../config/parallel_polygon_tasks_for_ht.py',
                                                  '../config/l2_random_random_test_model_3.py',
                                                  '../config/polygon_task.py')):
    """
    seeing if dropping number of filters decreases performance of best random l2
    yes, 10% to to 74.5%
    
    """
    protocols.extract_and_evaluate_protocol('../config/parallel_polygon_tasks_for_ht.py',
                                            '../config/l2_random_random_test_model_3.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='numpy', write=True,parallel=True)                                         