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
    47.60, 34.00, 40.86, 42.67, 2.40     overall
    
       
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
    
    46.87, 29.40, 40.39, 42.60, 3.15
    
    level 1, filter, norients
    value: [4, 6, 8]
    max: 45.20, 46.73, 46.87
    mean: 39.75, 40.35, 41.06
    quartiles 40.07, 40.60, 41.43
    
    level 1, filter, ker_shape
    value: [5, 7, 9, 11, 13, 17]
    max: 44.67, 46.73, 46.87, 45.47, 45.93, 42.53
    mean: 40.93, 41.11, 41.87, 40.90, 40.20, 37.30
    quartiles 41.80, 41.63, 42.23, 41.47, 40.40, 37.77
    
    level 2, filter, ker_shape
    value: [3, 5]
    max: 45.13, 46.87
    mean: 39.54, 41.23
    quartiles 39.87, 41.57
    
    level 2, filter, num_filters
    value: [128, 256, 384]
    max: 42.67, 46.87, 46.73
    mean: 37.40, 40.99, 42.77
    quartiles 37.37, 41.33, 43.03
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
    level 1, filter, norients
    value: [4, 6, 8]
    max: 94.20, 93.80, 93.90
    mean: 85.73, 83.69, 82.48
    quartiles 86.35, 84.40, 82.65
    
    level 1, filter, ker_shape
    value: [5, 7, 9, 11, 13, 17]
    max: 84.00, 88.10, 91.70, 93.10, 93.80, 94.20
    mean: 76.68, 81.03, 83.62, 86.01, 87.70, 88.78
    quartiles 77.25, 82.60, 84.55, 86.65, 88.35, 89.95
    
    level 2, filter, num_filters
    value: [128, 256, 384]
    max: 88.30, 93.50, 94.20
    mean: 78.89, 85.27, 87.74
    quartiles 79.00, 86.05, 88.45
    
    level 2, filter, ker_shape
    value: [3, 5]
    max: 93.90, 94.20
    mean: 84.72, 83.22
    quartiles 85.40, 83.35

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
    
    92.70, 61.00, 78.08, 83.60, 7.32
    
    level 2, filter, num_filters
    value: [40, 105, 216]
    max: 77.50, 86.00, 92.70
    mean: 70.37, 79.08, 84.79
    quartiles 70.75, 80.10, 84.95
    
    level 1, filter, ker_shape
    value: [5, 7, 9, 11, 13, 17]
    max: 80.10, 86.40, 87.30, 87.80, 90.00, 92.70
    mean: 71.88, 75.37, 78.34, 80.45, 79.67, 82.78
    quartiles 72.95, 75.70, 80.35, 80.10, 80.80, 83.70    

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
    85.20, 62.50, 76.38, 80.62, 5.04
    
    level 1, filter, num_filters
    value: [16, 30, 48]
    max: 83.90, 84.10, 85.20
    mean: 76.15, 76.35, 76.64
    quartiles 77.45, 77.45, 76.80
    
    level 1, filter, ker_shape
    value: [5, 7, 9, 11, 13, 17]
    max: 84.10, 84.10, 85.20, 83.10, 82.90, 80.00
    mean: 77.36, 77.59, 77.97, 76.57, 75.86, 72.92
    quartiles 78.40, 80.05, 79.40, 77.85, 76.05, 74.00
    
    level 2, filter, ker_shape
    value: [3, 5]
    max: 85.20, 83.50
    mean: 77.07, 75.69
    quartiles 78.25, 76.60
    
    level 2, filter, num_filters
    value: [128, 256, 384]
    max: 75.40, 82.90, 85.20
    mean: 70.50, 78.03, 80.61
    quartiles 70.55, 78.00, 81.00

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
    
    45.20, 27.67, 36.33, 38.87, 3.67    
    
    level 1, filter, num_filters
    value: [16, 30, 48]
    max: 43.27, 44.07, 45.20
    mean: 35.67, 36.67, 36.65
    quartiles 35.80, 36.10, 36.37
    
    level 1, filter, ker_shape
    value: [5, 7, 9, 11, 13, 17]
    max: 44.60, 44.53, 45.20, 43.00, 39.47, 39.87
    mean: 38.96, 38.13, 37.10, 35.75, 34.42, 33.62
    quartiles 39.20, 37.87, 37.30, 35.80, 34.50, 34.00
    
    level 2, filter, ker_shape
    value: [3, 5]
    max: 41.80, 45.20
    mean: 35.09, 37.57
    quartiles 35.00, 37.33
    
    level 2, filter, num_filters
    value: [128, 256, 384]
    max: 39.27, 43.40, 45.20
    mean: 33.21, 36.81, 38.97
    quartiles 33.53, 36.20, 38.87

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
    82.00, 55.00, 69.53, 76.35, 7.28
    
    level 2, filter, num_filters
    value: [40, 105, 216]
    max: 66.70, 76.00, 82.00
    mean: 61.86, 69.23, 77.49
    quartiles 62.65, 69.70, 78.10
    
    level 1, filter, ker_shape
    value: [5, 7, 9, 11, 13, 17]
    max: 82.00, 81.20, 80.10, 79.80, 78.90, 78.10
    mean: 70.39, 72.30, 70.98, 69.61, 68.48, 65.40
    quartiles 69.70, 72.70, 69.90, 67.00, 70.45, 66.00
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
    92.40, 70.90, 83.64, 88.83, 6.24
    
    level 1, filter, norients
    value: [4, 6, 8]  =  [16/40 , 30/105, 48/216  L1/L2 filters]
    max: 83.40, 90.20, 92.40
    mean: 78.12, 85.73, 87.07
    quartiles 78.45, 87.35, 89.00
    
    level 1, filter, ker_shape
    value: [5, 7, 9, 11, 13, 17]
    max: 80.50, 90.00, 92.40, 91.50, 91.10, 91.30
    mean: 74.31, 83.09, 85.85, 86.32, 85.96, 86.33
    quartiles 73.40, 83.95, 88.25, 87.20, 86.70, 88.65

    level 2, filter, ker_shape
    value: [3, 5]
    max: 92.40, 90.80
    mean: 85.01, 82.27
    quartiles 86.85, 83.70

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
    
    49.33, 30.93, 40.81, 43.10, 3.32
    
    level 1, filter, norients
    value: [4, 6, 8]
    max: 48.13, 47.27, 49.33
    mean: 40.18, 41.13, 41.11
    quartiles 40.13, 41.13, 41.00
    
    level 1, filter, ker_shape
    value: [5, 7, 9, 11, 13, 17]
    max: 49.33, 48.13, 46.73, 45.73, 44.00, 41.87
    mean: 43.16, 42.39, 41.68, 40.96, 39.32, 37.32
    quartiles 42.97, 42.60, 41.37, 41.17, 39.83, 37.77
        
    level 2, filter, ker_shape
    value: [3, 5]
    max: 47.00, 49.33
    mean: 40.42, 41.20
    quartiles 40.37, 41.10
    
    level 2, filter, num_filters
    value: [128, 256, 384]
    max: 43.40, 46.33, 49.33
    mean: 38.12, 41.06, 43.24
    quartiles 38.23, 41.27, 43.90
    """
    protocols.extract_and_evaluate_protocol('../config/renderman_tasks_for_ht_overlap.py',
                                            '../config/ht_l2_gabor_random_activation_ranges_models.py',
                                            '../config/ten_categories_images.py',
                                            convolve_func_name='numpy', write=True, parallel='semi')



#gabor random o2
@protocolize()
def make_ht_l2_gabor_and_random_models(depends_on='../config/ht_l2_gabor_and_random_models.py'):
    """
  
    """
    protocols.model_protocol(depends_on,parallel=False,write=True)
 

#gabor random o2 on renderman fav
@protocolize()
def ext_eval_ht_l2_gabor_and_random_fav(depends_on=('../config/fav_task.py',
                                                  '../config/ht_l2_gabor_and_random_models.py',
                                                  '../config/ten_categories_images.py')):
    """
    """
    a,b,c = depends_on
    protocols.extract_and_evaluate_protocol(a,b,c,convolve_func_name='numpy', write=True, parallel='semi')
                                            


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


@protocolize()
def make_ht_l2_random_random_o2_top5_renderman_models(depends_on='../config/ht_l2_random_random_o2_top5_renderman_models.py'):
    """
  
    """
    protocols.model_protocol(depends_on,parallel=False,write=True)


@protocolize()
def ext_eval_ht_l2_random_random_o2_top5_renderman_mixedup_renderman(depends_on=('../config/renderman_mixedup_alltasks.py',
                                                  '../config/ht_l2_random_random_o2_top5_renderman_models.py',
                                                  '../config/ten_categories_images.py')):
    """
    10 way control all: 41.266667 39.266667 39.786667 0.759415 
    10 way control trans: 89.000000 86.466667 88.066667 0.884433 
    10 way control inrot: 50.666667 46.933333 49.213333 1.729432 
    10 way mixedup all: 22.000000 19.066667 20.933333 1.056619 
    10 way mixedup trans: 74.800000 68.866667 71.120000 2.018316 
    10 way mixedup inrot: 31.066667 26.733333 29.026667 1.750987 
    carpln control all: 80.333333 71.000000 76.533333 3.159465 
    carpln control trans: 97.000000 92.000000 94.666667 1.873796 
    carpln control inrot: 89.000000 83.000000 86.400000 2.004440 
    carpln mixedup all: 60.333333 55.666667 58.333333 1.520234 
    carpln mixedup trans: 88.333333 83.666667 86.466667 1.939072 
    carpln mixedup inrot: 69.000000 66.000000 67.533333 1.127436 
    fav control all: 61.333333 59.111111 59.644444 0.850272 
    fav control trans: 91.555556 89.111111 90.222222 0.888889 
    fav control inrot: 57.333333 53.777778 55.600000 1.516331 
    fav mixedup all: 43.777778 34.666667 39.600000 2.898786 
    fav mixedup trans: 67.555556 62.666667 64.355556 1.857118 
    fav mixedup inrot: 47.777778 36.444444 41.111111 3.887301 
        
    """
    a,b,c = depends_on
    protocols.extract_and_evaluate_protocol(a,b,c,convolve_func_name='numpy', write=True, parallel='semi')

@protocolize()
def make_ht_l1_gabor_top5_renderman_models(depends_on='../config/ht_l1_gabor_top5_renderman_models.py'):
    """
  
    """
    protocols.model_protocol(depends_on,parallel=False,write=True)
    
@protocolize()
def ext_eval_ht_l1_gabor_top5_renderman_mixedup_renderman(depends_on=('../config/renderman_mixedup_alltasks.py',
                                                  '../config/ht_l1_gabor_top5_renderman_models.py',
                                                  '../config/ten_categories_images.py')):
    """
    10 way control all: 34.400000 31.800000 33.146667 0.869636 
    10 way control trans: 68.400000 64.533333 66.453333 1.501051 
    10 way control inrot: 41.133333 38.666667 39.533333 0.840106 
    10 way mixedup all: 16.133333 12.600000 14.960000 1.306803 
    10 way mixedup trans: 40.800000 33.866667 37.653333 2.927084 
    10 way mixedup inrot: 20.133333 17.600000 18.853333 0.890842 
    carpln control all: 73.666667 68.666667 71.133333 1.808621 
    carpln control trans: 89.000000 84.000000 85.800000 1.916014 
    carpln control inrot: 78.000000 74.000000 75.933333 1.583246 
    carpln mixedup all: 57.666667 50.000000 53.866667 3.138294 
    carpln mixedup trans: 71.333333 65.666667 68.933333 2.164358 
    carpln mixedup inrot: 62.000000 56.333333 59.333333 2.086997 
    fav control all: 47.777778 43.333333 45.333333 1.709812 
    fav control trans: 68.666667 63.777778 65.866667 1.753163 
    fav control inrot: 52.444444 48.222222 49.955556 1.641439 
    fav mixedup all: 38.888889 33.111111 36.444444 2.140728 
    fav mixedup trans: 57.111111 50.888889 53.377778 2.183666 
    fav mixedup inrot: 42.666667 39.333333 41.155556 1.328881 
    """
    a,b,c = depends_on
    protocols.extract_and_evaluate_protocol(a,b,c,convolve_func_name='numpy', write=True, parallel='semi')    

@protocolize()
def make_ht_l1_random_top5_polygon_models(depends_on='../config/ht_l1_random_top5_polygon_models.py'):
    """
  
    """
    protocols.model_protocol(depends_on,parallel=False,write=True)    

@protocolize()
def ext_eval_ht_l1_random_top5_polygon_mixedup_renderman(depends_on=('../config/renderman_mixedup_alltasks.py',
                                                  '../config/ht_l1_random_top5_polygon_models.py',
                                                  '../config/ten_categories_images.py')):
    """
    10 way control all: 30.466667 26.666667 28.866667 1.257157 
    10 way control trans: 66.733333 59.000000 63.533333 2.594010 
    10 way control inrot: 36.266667 34.133333 35.266667 0.747143 
    10 way mixedup all: 16.066667 13.733333 15.226667 0.866769 
    10 way mixedup trans: 40.066667 32.733333 36.520000 3.046718 
    10 way mixedup inrot: 20.466667 15.866667 18.053333 1.665013 
    carpln control all: 71.333333 65.000000 67.866667 2.526306 
    carpln control trans: 91.333333 85.333333 88.866667 2.166410 
    carpln control inrot: 76.333333 67.666667 72.733333 2.961981 
    carpln mixedup all: 53.666667 48.000000 51.400000 1.959592 
    carpln mixedup trans: 86.333333 64.333333 74.866667 7.476184 
    carpln mixedup inrot: 64.666667 53.000000 58.000000 3.983298 
    fav control all: 53.777778 46.666667 50.444444 2.330686 
    fav control trans: 79.111111 65.333333 73.333333 4.464400 
    fav control inrot: 44.666667 38.666667 42.844444 2.128699 
    fav mixedup all: 40.888889 32.888889 37.688889 3.249615 
    fav mixedup trans: 50.666667 44.000000 47.377778 2.201683 
    fav mixedup inrot: 42.888889 35.333333 38.977778 2.889231 
    """
    a,b,c = depends_on
    protocols.extract_and_evaluate_protocol(a,b,c,convolve_func_name='numpy', write=True, parallel='semi')  
    


##############################################
###############varied rotations###############

@Applies(deploy.images,args=('../config/ten_categories_rotated_images.py',True))
def generate_rotated_images():

    Apply()
    
    
@Applies(deploy.images,args=('../config/ten_categories_unrotated_images.py',True))
def generate_unrotated_images():
    Apply()
    
    
@protocolize()
def ext_eval_ht_l2_gabor_random_o2_top5_renderman_rotated_renderman(depends_on=('../config/renderman_tasks_for_ht_overlap.py',
                                                  '../config/ht_l2_gabor_random_o2_top5_renderman_models.py',
                                                  '../config/ten_categories_rotated_images.py')):
    """
    82.53, 77.93, 80.71, 81.93, 1.59   what?? 
    """
    a,b,c = depends_on
    protocols.extract_and_evaluate_protocol(a,b,c,convolve_func_name='numpy', write=True, parallel='semi')
    

@protocolize()
def ext_eval_ht_l2_gabor_random_o2_top5_renderman_unrotated_renderman(depends_on=('../config/renderman_tasks_for_ht_overlap.py',
                                                  '../config/ht_l2_gabor_random_o2_top5_renderman_models.py',
                                                  '../config/ten_categories_unrotated_images.py')):
    """
    87.60, 85.07, 85.91, 86.27, 0.94
    """
    a,b,c = depends_on
    protocols.extract_and_evaluate_protocol(a,b,c,convolve_func_name='numpy', write=True, parallel='semi')
    

@protocolize()
def ext_eval_ht_l2_gabor_random_o2_top5_renderman_mixedup_rotated_renderman(depends_on=('../config/renderman_mixedup_tasks.py',
                                                  '../config/ht_l2_random_random_o2_top5_renderman_models.py',
                                                  '../config/ten_categories_rotated_images.py')):
    """
      
    """
    a,b,c = depends_on
    protocols.extract_and_evaluate_protocol(a,b,c,convolve_func_name='numpy', write=True, parallel='semi')
  

    
############################################
###############pairwise tasks###############

@protocolize()
def ext_eval_ht_l2_gabor_random_o2_top5_pairwise_renderman(depends_on=('../config/renderman_pairwise_tasks.py',
                                                  '../config/ht_l2_gabor_random_o2_top5_renderman_models.py',
                                                  '../config/ten_categories_images.py')):
    """

    """
    a,b,c = depends_on
    protocols.extract_and_evaluate_protocol(a,b,c,convolve_func_name='numpy', write=True, parallel='semi')



#########################################
###############many levels###############

@protocolize()
def make_fourlevel_test_model(depends_on='../config/fourlevel_test_model.py'):
    """
    """
    protocols.model_protocol(depends_on,parallel=False,write=True)    

@protocolize()
def ext_eval_fourlevel_test_model_renderman(depends_on=('../config/renderman_tasks_for_ht_overlap.py',
                                                  '../config/fourlevel_test_model.py',
                                                  '../config/ten_categories_images.py')):
    """
    """
    a,b,c = depends_on
    protocols.extract_and_evaluate_protocol(a,b,c,convolve_func_name='numpy', write=True, parallel='semi')  


@protocolize()
def make_fourlevel_test_model_2(depends_on='../config/fourlevel_test_model_2.py'):
    """
    """
    protocols.model_protocol(depends_on,parallel=False,write=True)    

@protocolize()
def ext_eval_fourlevel_test_model_2_renderman(depends_on=('../config/renderman_tasks_for_ht_overlap.py',
                                                  '../config/fourlevel_test_model_2.py',
                                                  '../config/ten_categories_images.py')):
    """
    """
    a,b,c = depends_on
    protocols.extract_and_evaluate_protocol(a,b,c,convolve_func_name='numpy', write=True, parallel='semi')  


@protocolize()
def make_fourlevel_test_model_3(depends_on='../config/fourlevel_test_model_3.py'):
    """
    """
    protocols.model_protocol(depends_on,parallel=False,write=True)    

@protocolize()
def ext_eval_fourlevel_test_model_3_renderman(depends_on=('../config/renderman_tasks_for_ht_overlap.py',
                                                  '../config/fourlevel_test_model_3.py',
                                                  '../config/ten_categories_images.py')):
    """
    """
    a,b,c = depends_on
    protocols.extract_and_evaluate_protocol(a,b,c,convolve_func_name='numpy', write=True, parallel='semi')  


@protocolize()
def make_fourlevel_test_model_4(depends_on='../config/fourlevel_test_model_4.py'):
    """
    """
    protocols.model_protocol(depends_on,parallel=False,write=True)    

@protocolize()
def ext_eval_fourlevel_test_model_4_renderman(depends_on=('../config/renderman_tasks_for_ht_overlap.py',
                                                  '../config/fourlevel_test_model_4.py',
                                                  '../config/ten_categories_images.py')):
    """
    """
    a,b,c = depends_on
    protocols.extract_and_evaluate_protocol(a,b,c,convolve_func_name='numpy', write=True, parallel='semi')  

@protocolize()
def make_fourlevel_test_model_5(depends_on='../config/fourlevel_test_model_5.py'):
    """
    """
    protocols.model_protocol(depends_on,parallel=False,write=True)    

@protocolize()
def ext_eval_fourlevel_test_model_5_renderman(depends_on=('../config/renderman_tasks_for_ht_overlap.py',
                                                  '../config/fourlevel_test_model_5.py',
                                                  '../config/ten_categories_images.py')):
    """
    """
    a,b,c = depends_on
    protocols.extract_and_evaluate_protocol(a,b,c,convolve_func_name='numpy', write=True, parallel='semi')  

@protocolize()
def make_fourlevel_test_model_6(depends_on='../config/fourlevel_test_model_6.py'):
    """
    """
    protocols.model_protocol(depends_on,parallel=False,write=True)    

@protocolize()
def ext_eval_fourlevel_test_model_6_renderman(depends_on=('../config/renderman_tasks_for_ht_overlap.py',
                                                  '../config/fourlevel_test_model_6.py',
                                                  '../config/ten_categories_images.py')):
    """
    """
    a,b,c = depends_on
    protocols.extract_and_evaluate_protocol(a,b,c,convolve_func_name='numpy', write=True, parallel='semi')  


@protocolize()
def make_fourlevel_test_model_7(depends_on='../config/fourlevel_test_model_7.py'):
    """
    """
    protocols.model_protocol(depends_on,parallel=False,write=True)    

@protocolize()
def ext_eval_fourlevel_test_model_7_renderman(depends_on=('../config/renderman_tasks_for_ht_overlap.py',
                                                  '../config/fourlevel_test_model_7.py',
                                                  '../config/ten_categories_images.py')):
    """
    """
    a,b,c = depends_on
    protocols.extract_and_evaluate_protocol(a,b,c,convolve_func_name='numpy', write=True, parallel='semi')  
    
@protocolize()
def make_ht_manylevel_random_models(depends_on='../config/ht_manylevel_random_models.py'):
    """
    """
    protocols.model_protocol(depends_on,parallel=False,write=True)    

@protocolize()
def ext_eval_ht_manylevel_random_models_renderman(depends_on=('../config/renderman_tasks_for_ht_overlap.py',
                                                  '../config/ht_manylevel_random_models.py',
                                                  '../config/ten_categories_images.py')):
    """
    """
    a,b,c = depends_on
    protocols.extract_and_evaluate_protocol(a,b,c,convolve_func_name='numpy', write=True, parallel='semi')      
    

@protocolize()
def make_ht_manylevel_random_models_2(depends_on='../config/ht_manylevel_random_models_2.py'):
    """
    """
    protocols.model_protocol(depends_on,parallel=False,write=True)    

@protocolize()
def ext_eval_ht_manylevel_random_models_2_renderman(depends_on=('../config/renderman_tasks_for_ht_overlap2.py',
                                                  '../config/ht_manylevel_random_models_2.py',
                                                  '../config/ten_categories_images.py')):
    """
    """
    a,b,c = depends_on
    protocols.extract_and_evaluate_protocol(a,b,c,convolve_func_name='numpy', write=True, parallel='semi')          
    

@protocolize()
def make_ht_manylevel_random_scale_comparison_models(depends_on='../config/ht_manylevel_random_scale_comparison_models.py'):
    """
    """
    protocols.model_protocol(depends_on,parallel=False,write=True)    

@protocolize()
def ext_eval_ht_manylevel_random_scale_comparison_models_renderman(depends_on=('../config/renderman_tasks_for_ht_overlap.py',
                                                  '../config/ht_manylevel_random_scale_comparison_models.py',
                                                  '../config/ten_categories_images.py')):
    """
    """
    a,b,c = depends_on
    protocols.extract_and_evaluate_protocol(a,b,c,convolve_func_name='numpy', write=True, parallel='semi')              
    
    
##########################################   
#################3d gabor#################

@protocolize()
def make_lt_l2_gabor_gabor_models(depends_on='../config/lt_l2_gabor_gabor_models.py'):
    """
    """
    protocols.model_protocol(depends_on,parallel=False,write=True)    

@protocolize()
def ext_eval_lt_l2_gabor_gabor_models_renderman(depends_on=('../config/renderman_tasks_for_ht_overlap.py',
                                                  '../config/lt_l2_gabor_gabor_models.py',
                                                  '../config/ten_categories_images.py')):
    """
    """
    a,b,c = depends_on
    protocols.extract_and_evaluate_protocol(a,b,c,convolve_func_name='numpy', write=True, parallel='semi')              
    


@protocolize()
def make_lt_l2_gabor_gabor_models2(depends_on='../config/lt_l2_gabor_gabor_models2.py'):
    """
    """
    protocols.model_protocol(depends_on,parallel=False,write=True)    

@protocolize()
def ext_eval_lt_l2_gabor_gabor_models2_renderman(depends_on=('../config/renderman_tasks_for_ht_overlap.py',
                                                  '../config/lt_l2_gabor_gabor_models2.py',
                                                  '../config/ten_categories_images.py')):
    """
    """
    a,b,c = depends_on
    protocols.extract_and_evaluate_protocol(a,b,c,convolve_func_name='numpy', write=True, parallel='semi')    
    
@protocolize()
def make_lt_l2_gabor_gabor_models3(depends_on='../config/lt_l2_gabor_gabor_models3.py'):
    """
    """
    protocols.model_protocol(depends_on,parallel=False,write=True)    

@protocolize()
def ext_eval_lt_l2_gabor_gabor_models3_renderman(depends_on=('../config/renderman_tasks_for_ht_overlap.py',
                                                  '../config/lt_l2_gabor_gabor_models3.py',
                                                  '../config/ten_categories_images.py')):
    """
    """
    a,b,c = depends_on
    protocols.extract_and_evaluate_protocol(a,b,c,convolve_func_name='numpy', write=True, parallel='semi')        