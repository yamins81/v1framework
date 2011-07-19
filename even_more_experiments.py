from starflow.protocols import Apply, Applies, protocolize

import deploy
import pythor_protocols as protocols



@protocolize()
def make_ht_l2_gabor_random_squared_models(depends_on='../config/ht_l2_gabor_random_squared_models.py'):
    """
  
    """
    protocols.model_protocol(depends_on,parallel=False,write=True)




@protocolize()
def ext_eval_ht_l2_gabor_random_squared_renderman(depends_on=('../config/renderman_tasks_for_ht2.py',
                                                  '../config/ht_l2_gabor_random_squared_models.py',
                                                  '../config/ten_categories_images.py')):
    """
    47.60, 34.00, 40.86, 42.67, 2.40	 overall
    
       
    about a 2% improvement over direct comparison, e.g. 
    	44.27, 38.27, 42.15, 43.30, 1.74  (this run with l0 norm shape = 9 and ker_size < 21 -- this is a small sample)
    	47.60, 34.00, 41.23, 42.97, 2.34  (this run with just with l1 ker_shape < 21 restriction)
    vs.
    	47.47, 32.27, 39.89, 41.47, 2.67  (gabor/random l2 with 384 l2 filters with ker_shape > 5)
    
    clearly increases performance -- esp. seen in quartile -- on various combinations:
    	l1 filter ker in [7,13]:
    		47.60, 38.40, 42.40, 44.20, 2.36    for 7 & 13
         	46.27, 39.60, 43.08, 44.40, 1.65    for 7 & 9
         	46.27, 39.47, 42.69, 43.73, 1.81    for 7 & 11
         	46.27, 36.80, 41.20, 43.13, 2.38    for 7 & 17
         	46.27, 34.93, 40.49, 42.67, 2.98    for 7 & 21
         	
    		44.40, 38.13, 41.07, 41.87, 1.73    for just 7
    		44.13, 38.00, 41.49, 42.70, 1.69    for just 9
    		44.13, 37.87, 40.36, 41.63, 1.96    for just 11
    		47.47, 37.33, 39.82, 41.03, 2.67    for just 13
    		44.40, 38.13, 41.07, 41.87, 1.73    for just 17
         	
    
         	
    """
    protocols.extract_and_evaluate_protocol('../config/renderman_tasks_for_ht2.py',
                                            '../config/ht_l2_gabor_random_squared_models.py',
                                            '../config/ten_categories_images.py',
                                            convolve_func_name='numpy', write=True,parallel=True)



@protocolize()
def make_ht_l2_gabor_random_bothactivationranges_models(depends_on='../config/ht_l2_gabor_random_bothactivationranges_models.py'):
    """
  
    """
    protocols.model_protocol(depends_on,parallel=False,write=True)
 

@protocolize()
def ext_eval_ht_l2_gabor_random_bothactivationranges_renderman(depends_on=('../config/renderman_tasks_for_ht3.py',
                                                  '../config/ht_l2_gabor_random_bothactivationranges_models.py',
                                                  '../config/ten_categories_images.py')):
    """
    44.87, 31.93, 38.23, 41.13, 3.35    
            --> comparison is 
                (41.333333333333329, 38.000000000000007, 40.0, 1.225651754056678)

    """
    protocols.extract_and_evaluate_protocol('../config/renderman_tasks_for_ht3.py',
                                            '../config/ht_l2_gabor_random_bothactivationranges_models.py',
                                            '../config/ten_categories_images.py',
                                            convolve_func_name='numpy', write=True,parallel=True)

######
######
######

#gabor random o2
@protocolize()
def make_ht_l2_gabor_random_o2_models(depends_on='../config/ht_l2_gabor_random_o2_models.py'):
    """
  
    """
    protocols.model_protocol(depends_on,parallel=False,write=True)
 

#gabor random o2 on renderman
@protocolize()
def ext_eval_ht_l2_gabor_random_o2_renderman(depends_on=('../config/renderman_tasks_for_ht_overlap.py',
                                                  '../config/ht_l2_gabor_random_o2_models.py',
                                                  '../config/ten_categories_images.py')):
    """

    """
    protocols.extract_and_evaluate_protocol('../config/renderman_tasks_for_ht_overlap.py',
                                            '../config/ht_l2_gabor_random_o2_models.py',
                                            '../config/ten_categories_images.py',
                                            convolve_func_name='numpy', write=True, parallel='semi')
                                            
                                                                                        

#gabor random o2 on polygon
@protocolize()
def ext_eval_ht_l2_gabor_random_o2_polygon(depends_on=('../config/parallel_polygon_tasks_for_ht_overlap.py',
                                                  '../config/ht_l2_gabor_random_o2_models.py',
                                                  '../config/polygon_task.py')):
    """

    """
    protocols.extract_and_evaluate_protocol('../config/parallel_polygon_tasks_for_ht_overlap.py',
                                            '../config/ht_l2_gabor_random_o2_models.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='numpy', write=True, parallel='semi')

#gabor random fewer filters to balance freq model
@protocolize()
def make_ht_l2_gabor_random_o2_fewer_filter_models(depends_on='../config/ht_l2_gabor_random_o2_fewer_filter_models.py'):
    """
  
    """
    protocols.model_protocol(depends_on,parallel=False,write=True)
 
 
#gabor random o2 fewer_filter on polygon
@protocolize()
def ext_eval_ht_l2_gabor_random_o2_fewer_filter_polygon(depends_on=('../config/parallel_polygon_tasks_for_ht_overlap.py',
                                                  '../config/ht_l2_gabor_random_o2_fewer_filter_models.py',
                                                  '../config/polygon_task.py')):
    """

    """
    a,b,c = depends_on
    protocols.extract_and_evaluate_protocol(a,b,c,convolve_func_name='numpy', write=True, parallel='semi')


#random random o2
@protocolize()
def make_ht_l2_random_random_o2_models(depends_on='../config/ht_l2_random_random_o2_models.py'):
    """
  
    """
    protocols.model_protocol(depends_on,parallel=False,write=True)
 
 
#random random o2 on polygon
@protocolize()
def ext_eval_ht_l2_random_random_o2_polygon(depends_on=('../config/parallel_polygon_tasks_for_ht_overlap.py',
                                                  '../config/ht_l2_random_random_o2_models.py',
                                                  '../config/polygon_task.py')):
    """

    """
    protocols.extract_and_evaluate_protocol('../config/parallel_polygon_tasks_for_ht_overlap.py',
                                            '../config/ht_l2_random_random_o2_models.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='numpy', write=True, parallel='semi')

#random random o2 on renderman
@protocolize()
def ext_eval_ht_l2_random_random_o2_renderman(depends_on=('../config/renderman_tasks_for_ht_overlap.py',
                                                  '../config/ht_l2_random_random_o2_models.py',
                                                  '../config/ten_categories_images.py')):
    """

    """
    protocols.extract_and_evaluate_protocol('../config/renderman_tasks_for_ht_overlap.py',
                                            '../config/ht_l2_random_random_o2_models.py',
                                            '../config/ten_categories_images.py',
                                            convolve_func_name='numpy', write=True, parallel='semi')

#random random fewer filters 
@protocolize()
def make_ht_l2_random_random_o2_fewer_filter_models(depends_on='../config/ht_l2_random_random_o2_fewer_filter_models.py'):
    """
  
    """
    protocols.model_protocol(depends_on,parallel=False,write=True)
 
 
#random random o2 fewer_filter on polygon
@protocolize()
def ext_eval_ht_l2_random_random_o2_fewer_filter_polygon(depends_on=('../config/parallel_polygon_tasks_for_ht_overlap.py',
                                                  '../config/ht_l2_random_random_o2_fewer_filter_models.py',
                                                  '../config/polygon_task.py')):
    """

    """
    a,b,c = depends_on
    protocols.extract_and_evaluate_protocol(a,b,c,convolve_func_name='numpy', write=True, parallel='semi')


#gabor freq uniform
@protocolize()
def make_ht_l2_gabor_freq_models(depends_on='../config/ht_l2_freq_models_3.py'):
    """
  
    """
    protocols.model_protocol(depends_on,parallel=False,write=True)
 
 
#gabor freq uniform on polygon 
@protocolize()
def ext_eval_ht_l2_gabor_freq_polygon(depends_on=('../config/parallel_polygon_tasks_for_ht_overlap.py',
                                                  '../config/ht_l2_freq_models_3.py',
                                                  '../config/polygon_task.py')):
    """

    """
    protocols.extract_and_evaluate_protocol('../config/parallel_polygon_tasks_for_ht_overlap.py',
                                            '../config/ht_l2_freq_models_3.py',
                                            '../config/polygon_task.py',
                                            convolve_func_name='numpy', write=True, parallel='semi')




#gabor random activation ranges
@protocolize()
def make_ht_l2_gabor_random_activation_ranges_models(depends_on='../config/ht_l2_gabor_random_activation_ranges_models.py'):
    """
  
    """
    protocols.model_protocol(depends_on,parallel=False,write=True)
 

#gabor random activation ranges on renderman
@protocolize()
def ext_eval_ht_l2_gabor_random_activation_ranges_renderman(depends_on=('../config/renderman_tasks_for_ht_overlap.py',
                                                  '../config/ht_l2_gabor_random_activation_ranges_models.py',
                                                  '../config/ten_categories_images.py')):
    """

    """
    protocols.extract_and_evaluate_protocol('../config/renderman_tasks_for_ht_overlap.py',
                                            '../config/ht_l2_gabor_random_activation_ranges_models.py',
                                            '../config/ten_categories_images.py',
                                            convolve_func_name='numpy', write=True, parallel='semi')



############mixed up tasks#########


@protocolize()
def make_ht_l2_gabor_random_o2_top5_renderman_models(depends_on='../config/ht_l2_gabor_random_o2_top5_renderman_models.py'):
    """
  
    """
    protocols.model_protocol(depends_on,parallel=False,write=True)
 

@protocolize()
def ext_eval_ht_l2_gabor_random_o2_top5_renderman_mixedup_renderman(depends_on=('../config/renderman_mixedup_tasks.py',
                                                  '../config/ht_l2_gabor_random_o2_top5_renderman_models.py',
                                                  '../config/ten_categories_images.py')):
    """
    10 way control: (43.399999999999999, 40.199999999999996, 41.546666666666667, 1.0802468853615528)
    10 way mixed up: (22.266666666666669, 19.333333333333336, 20.546666666666667, 0.97378984728054507)
    	--> fairy large drop, but mixed up is still 2x chance (10%)
    car/plane control: (80.0, 75.999999999999986, 77.599999999999994, 1.3727506854649372)
    car/plane mixed up: (50.0, 46.0, 47.600000000000001, 1.3727506854649327)
    	--> big drop, all the way to chance (50%)
    furn/anim/veh control: (56.000000000000014, 49.555555555555557, 53.066666666666677, 2.6006646823693877)
    furn/anim/veh mixed up: (39.55555555555555, 35.555555555555557, 37.555555555555557, 1.3626408637116323)
    	--> less big drop (since control is not great) but goes down close to chance (33%)
    """
    a,b,c = depends_on
    protocols.extract_and_evaluate_protocol(a,b,c,convolve_func_name='numpy', write=True, parallel='semi')


@protocolize()
def ext_eval_ht_l2_gabor_random_o2_top5_renderman_mixedup_sub_renderman(depends_on=('../config/renderman_mixedup_subtasks.py',
                                                  '../config/ht_l2_gabor_random_o2_top5_renderman_models.py',
                                                  '../config/ten_categories_images.py')):
    """
    10 way trans only control : (92.133333333333326, 90.599999999999994, 91.373333333333335, 0.64442911859170915)
    10 way inrot only control : (53.600000000000001, 48.93333333333333, 51.533333333333339, 1.9727026019031784)
    10 way trans only mixedup : (74.0, 72.0, 73.306666666666672, 0.78835975995170626)
    10 way inrot only mixedup : (30.466666666666669, 28.333333333333332, 29.32, 0.92989844128867782)
    carplane trans only control : (94.999999999999986, 93.666666666666657, 94.399999999999977, 0.44221663871405092)
    carplane inrot only control : (90.333333333333343, 86.333333333333343, 88.866666666666674, 1.4391355429948589)
    carplane trans only mixedup : (90.666666666666657, 82.333333333333343, 87.0, 2.6749870196985106)
    carplane inrot only mixedup : (72.333333333333343, 65.666666666666671, 68.533333333333331, 2.4908722256359219)
    fav trans only control : (86.666666666666671, 83.111111111111114, 84.62222222222222, 1.3655370218433061)
    fav inrot only control : (62.222222222222229, 58.666666666666671, 60.488888888888894, 1.2911856930074617)
    fav trans only mixedup : (73.111111111111114, 65.555555555555543, 69.288888888888877, 2.4073960113690438)
    fav inrot only mixedup : (44.444444444444436, 38.0, 41.066666666666663, 2.573943508545824)

    """
    a,b,c = depends_on
    protocols.extract_and_evaluate_protocol(a,b,c,convolve_func_name='numpy', write=True, parallel='semi')

