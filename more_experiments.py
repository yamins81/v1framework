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
    	
    cars vs planes:  
    just translation: (95.833333333333329, 86.666666666666671, 91.770833333333329, 2.6331718041176106)
    just in-plane rotation: (89.166666666666671, 71.666666666666671, 80.677083333333329, 5.222636609962656)
    just out of plane rotation: (82.5, 66.666666666666671, 76.09375, 4.504229782937367)
    just scaling: (94.166666666666671, 82.5, 90.260416666666671, 3.4575645480839059)
    scaling and in plane rotation: (79.166666666666671, 59.166666666666664, 68.333333333333343, 5.5277079839256666)
    translation and inplane rotation: (83.333333333333329, 67.5, 75.15625, 3.8552222653535515)
    all rotation: (79.166666666666671, 57.5, 69.947916666666671, 5.7317702417461467)

    chair vs table:  
    just translation: (94.166666666666671, 80.0, 88.072916666666657, 3.4575645480839063)
    just in-plane rotation: (73.333333333333329, 55.833333333333336, 63.437500000000007, 4.3188359253391413)
    just out of plane rotation: 
    just scaling: (91.666666666666671, 73.333333333333329, 83.229166666666671, 4.822140416292795)
    scaling and in plane rotation: (70.0, 50.833333333333336, 58.541666666666664, 6.4717602362537772)
    translation and inplane rotation: (63.333333333333336, 51.666666666666664, 58.59375, 3.1693895913794585)
    all rotation: (65.0, 50.833333333333336, 57.343749999999993, 4.5664342816481129)
    
    furniture vs animals vs vehicles:
    just translation: (91.666666666666671, 72.5, 80.572916666666671, 4.4068163544609442)
    just in-plane rotation: (57.5, 47.5, 52.447916666666664, 2.5424990098872757)
    just out of plane rotation: (74.166666666666671, 50.833333333333336, 60.572916666666679, 6.3102564082214601)
    just scaling: (85.0, 68.333333333333329, 75.78125, 4.7550115192932099)
    scaling and in plane rotation: (59.166666666666664, 41.666666666666664, 48.4375, 4.5821494683172439)
    translation and inplane rotation: (56.666666666666664, 43.333333333333336, 48.124999999999993, 3.5047091335959828)
    all rotation: (52.5, 35.833333333333336, 45.989583333333329, 4.3122358010858663)

    """
    protocols.extract_and_evaluate_protocol('../config/renderman_subtasks.py',
                                            '../config/various_random_l3_grayscale_models.py',
                                            '../config/ten_categories_images.py',
                                            convolve_func_name='numpy', write=True,parallel=True)

@protocolize()
def ext_eval_various_random_l3_renderman_subtasks_backgrounds(depends_on=('../config/renderman_subtasks_backgrounds.py',
                                                  '../config/various_random_l3_grayscale_models.py',
                                                  '../config/ten_categories_images.py')):
    """
    """
    protocols.extract_and_evaluate_protocol('../config/renderman_subtasks_backgrounds.py',
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
    
    cars vs planes:  
    just translation: (98.333333333333329, 86.666666666666671, 92.8888888888889, 2.7866524897743212)
    just in-plane rotation: (85.0, 70.0, 77.499999999999986, 3.9907299991262164)
    just out of plane rotation: (79.166666666666671, 65.0, 73.222222222222214, 3.7126743214674733)
    just scaling: (94.166666666666671, 87.5, 90.666666666666671, 1.8053418676968802)
    scaling and in plane rotation: (68.333333333333329, 55.833333333333336, 64.444444444444443, 3.2441780712569841)
    translation and inplane rotation: (83.333333333333329, 57.5, 69.944444444444443, 6.1948804031939577)
    all rotation: (75.833333333333329, 53.333333333333336, 67.222222222222214, 5.9447040441757473)

    chair vs table:  
    just translation: (90.833333333333329, 81.666666666666671, 86.111111111111128, 2.3700810008557278)
    just in-plane rotation: (61.666666666666664, 45.833333333333336, 55.111111111111121, 4.2368168810972477)
    just out of plane rotation: 
    just scaling: (95.0, 83.333333333333329, 91.222222222222229, 2.948110924760357)
    scaling and in plane rotation: (69.166666666666671, 48.333333333333336, 60.666666666666671, 5.9488560991915822)
    translation and inplane rotation: (68.333333333333329, 51.666666666666664, 61.444444444444443, 4.4562343622567724)
    all rotation: (65.0, 51.666666666666664, 59.111111111111114, 3.4756649599149019)
    
    furniture vs animals vs vehicles:
    just translation: (75.0, 65.833333333333329, 71.0, 2.7419917065006816)
    just in-plane rotation: (55.833333333333336, 40.0, 45.833333333333343, 3.9557740264262007)
    just out of plane rotation: (59.166666666666664, 40.0, 45.777777777777779, 4.7284272957673812)
    just scaling: (70.0, 58.333333333333336, 62.777777777777779, 3.4246744460938769)
    scaling and in plane rotation: (48.333333333333336, 35.833333333333336, 41.444444444444436, 3.6826252042614556)
    translation and inplane rotation: (55.0, 34.166666666666664, 43.777777777777779, 5.6344399526702666)
    all rotation: (46.666666666666664, 28.333333333333332, 40.222222222222221, 4.7284272957673803)
    
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
    (80.0, 61.299999999999997, 71.869444444444454, 4.3336743099466251)

    
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
    (84.099999999999994, 47.200000000000003, 73.85277777777776, 6.2112943473269571)
     -- performance averages for different size l1 filters: 
    	sizes: [5,7,9,11,13,17]
    	mean: [76.2, 77.1, 74.95, 73.5, 72.3, 69] 
   		max: [81.4, 84.1, 83.5, 81.2, 79.4, 79.1]
     -- l2 filter shape:
     	sizes: [3,5,7,9]
    	mean: [75.75, 75.4, 73.42, 70.838888888888903]
    	max: [83.7, 84.1, 81.2, 79.4]
     -- l1 norm shape:
     	sizes: 3,5,9
     	mean: [76.25, 75.402777777777771, 69.905555555555551]
     	max: [84.099999999999994, 82.099999999999994, 81.400000000000006])
     -- l1 pool shape: ([3,5,9])
        mean: [72.723611111111097, 76.394444444444431, 72.440277777777794]
        max: [77.900000000000006, 82.099999999999994, 84.099999999999994]
    
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
    overall:
        (94.799999999999997, 68.099999999999994, 82.516666666666708, 6.6660327079524651)
    with 32/256 filters:
    	(91.700000000000003, 71.200000000000003, 83.133333333333326, 5.1428483245074306)
    with 16/128 filters:
        (85.099999999999994, 68.099999999999994, 77.109523809523822, 4.6513597358503951)
    with 48/384 filters:
    	(94.799999999999997, 70.299999999999997, 87.307142857142836, 5.7147008777755781)
    	
    -- having more filters definitely helps
    -- performance averages for different size l1 filters: 
    	sizes: [5,7,9,11,13,17,21]
    	mean performances: [72.61, 81.75, 85.42, 86.1, 85.7, 84.5, 81.5]
    	max perforamnces: [78.9, 91.8, 94.8, 93.1, 93.5, 91.1, 90.1]
    -- l2 filters:
    	sizes: [3,5,7]
    	mean: [85.0, 82.5, 80.1]
    	max: [94.8, 92.1, 89.5]
    -- l1 norm shape almost no affect:
    	sizes: [3,5]
    	mean: [82.8, 82.2]
    	max: [94.8, 92.8]
    	
    -- l1 filters for 32/256 filters:
        sizes: [5,7,9,11,13,17,21]
    	mean: [73.4, 80.72, 87.3, 86.55, 87.0, 85.5, 81.4]
        max: [75.5, 83.8, 91.7, 89.1, 90.4, 90.0, 84.6]

    

    
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
    max,min,mean,std:
    
    (76.700000000000003, 48.100000000000001, 60.363580246913564, 5.5775648052838154)
    
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
    max,min,mean,std:
        (93.799999999999997, 47.100000000000001, 79.318981481481487, 9.3502802928689039)
    
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
    max,min,mean,std:
       (87.5, 49.100000000000001, 73.285714285714292, 9.1545089444276275)
    
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
                                            
@protocolize()
def make_l1_gabor_test_model(depends_on='../config/l1_gabor_test_model.py'):
    """

    """
    protocols.model_protocol(depends_on,parallel=False,write=True)
    

@protocolize()
def ext_eval_l1_gabor_test_polygon(depends_on=('../config/rect_task.py',
                                                  '../config/l1_gabor_test_model.py',
                                                  '../config/polygon_task.py')):
    """
    seeing if the best l1 gabors identified as good for the poly 5-way task does well on the rect task: yes. 87.5%
    
    """
    protocols.extract_and_evaluate_protocol('../config/rect_task.py',
                                            '../config/l1_gabor_test_model.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='numpy', write=True,parallel=True)                 
                                            
@protocolize()
def make_l1_gabor_test_model_2(depends_on='../config/l1_gabor_test_model_2.py'):
    """

    """
    protocols.model_protocol(depends_on,parallel=False,write=True)
    

@protocolize()
def ext_eval_l1_gabor_test_2_polygon(depends_on=('../config/parallel_polygon_tasks_for_ht.py',
                                                  '../config/l1_gabor_test_model_2.py',
                                                  '../config/polygon_task.py')):
    """
    seeing if adding more filters to the best l1 gabor helps -- yes a little, 87.8% (try more?)
    
    """
    protocols.extract_and_evaluate_protocol('../config/parallel_polygon_tasks_for_ht.py',
                                            '../config/l1_gabor_test_model_2.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='numpy', write=True,parallel=True)                                                                                                             
                                            
@protocolize()
def make_l1_gabor_test_model_3(depends_on='../config/l1_gabor_test_model_3.py'):
    """

    """
    protocols.model_protocol(depends_on,parallel=False,write=True)
    

@protocolize()
def ext_eval_l1_gabor_test_3_polygon(depends_on=('../config/parallel_polygon_tasks_for_ht.py',
                                                  '../config/l1_gabor_test_model_3.py',
                                                  '../config/polygon_task.py')):
    """
    seeing if adding even more filters helps ... 
    
    """
    protocols.extract_and_evaluate_protocol('../config/parallel_polygon_tasks_for_ht.py',
                                            '../config/l1_gabor_test_model_3.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='numpy', write=True,parallel=True)                                                                                                                                                         