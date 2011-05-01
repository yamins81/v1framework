#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import cPickle
import hashlib

import scipy as sp
import pymongo as pm
import gridfs
from bson import SON
import bson

from starflow.protocols import protocolize, actualize
from starflow.utils import activate

import v1like_extract as v1e
import v1like_funcs as v1f
import traintest
from v1like_extract import get_config
import svm

from dbutils import get_config_string, get_filename, reach_in, DBAdd, createCertificateDict, son_escape, do_initialization, get_most_recent_files


try:
    import v1_pyfft
except:
    GPU_SUPPORT = False
else:
    GPU_SUPPORT = True
    

###gridded gabors square vs. rect    

@protocolize()
def pull_gridded_gabors_sq_vs_rect_test(depends_on = '../config/config_pull_gridded_gabors_sq_vs_rect_test.py'):
    """basic test of pull protocol with gabor filters, low density transformation image set of squares versus rectangles
    Result: gabor filters do great"""
    D = v1_pull_protocol(depends_on,)
    actualize(D)


@protocolize()
def pull_gridded_gabors_sq_vs_rect(depends_on = '../config/config_pull_gridded_gabors_sq_vs_rect_evaluation.py'):
    """test of standard 96-filter gridded gabor filterbank on high density transformations set of squares versus rectangles
    Result: gabor filters do great"""
    D = v1_pull_protocol(depends_on)
    actualize(D)
    
@protocolize()
def pull_gridded_gabors_sq_vs_rect_mcc(depends_on = '../config/config_pull_gridded_gabors_sq_vs_rect_mcc.py'):
    """trying out maximum correlation classifier with 96 gridded gabors
    RESULT: """
    D = v1_pull_protocol(depends_on)
    actualize(D)    

@protocolize()
def pull_gridded_gabors_sq_vs_rect_two_orthogonal_mcc(depends_on = '../config/config_pull_gridded_gabors_sq_vs_rect_two_orthogonal_mcc.py'):
    """trying out maximum correlation classifier with two orthogonal gridded gabors that perform well with SVM
    RESULT: """
    D = v1_pull_protocol(depends_on)
    actualize(D)
    
@protocolize()
def pull_gridded_gabors_sq_vs_rect_one_filter_mcc(depends_on = '../config/config_pull_gridded_gabors_sq_vs_rect_one_filter_mcc.py'):
    """trying out maximum correlation classifier with two orthogonal gridded gabors that perform well with SVM
    RESULT: """
    D = v1_pull_protocol(depends_on)
    actualize(D)    
    
@protocolize()
def pull_gridded_gabors_sq_vs_rect_smallfilters(depends_on = '../config/config_pull_gridded_gabors_sq_vs_rect_smallfilters_evaluation.py'):
    """test of 48-filter gridded gabor filterbank on high density transformations set of squares versus rectangles
    Result: still great"""
    D = v1_pull_protocol(depends_on)
    actualize(D)    
    
    
@protocolize()
def pull_gridded_gabors_sq_vs_rect_verysmallfilters(depends_on = '../config/config_pull_gridded_gabors_sq_vs_rect_verysmallfilters_evaluation.py'):
    """test of 9-filter gridded gabor filterbank on high density transformations set of squares versus rectangles
    Result: STILL great"""
    D = v1_pull_protocol(depends_on)
    actualize(D)        
    
@protocolize()
def pull_gridded_gabors_sq_vs_rect_extremelysmallfilters(depends_on = '../config/config_pull_gridded_gabors_sq_vs_rect_extremelysmallfilters_evaluation.py'):
    """test of 4-filter gridded gabor filterbank on high density transformations set of squares versus rectangles
    Result: STILL quite good"""
    D = v1_pull_protocol(depends_on)
    actualize(D)         
    
@protocolize()
def pull_gridded_gabors_sq_vs_rect_veryveryfewfilters(depends_on = '../config/config_pull_gridded_gabors_sq_vs_rect_veryveryfewfilters_evaluation.py'):
    """test of two-filter gridded gabor filterbank on high density transformations set of squares versus rectangles
    Result: STILL pretty good"""
    D = v1_pull_protocol(depends_on)
    actualize(D)  
    
    
@protocolize()
def pull_gridded_gabors_sq_vs_rect_onefilter(depends_on = '../config/config_pull_gridded_gabors_sq_vs_rect_onefilter_evaluation.py'):
    """test of 1-filter 'gridded' gabor filterbank on high density transformations set of squares versus rectangles
    Result: not great""" 
    D = v1_pull_protocol(depends_on)
    actualize(D)
    
@protocolize()
def pull_gridded_gabors_sq_vs_rect_various_onefilter(depends_on = '../config/config_pull_gridded_gabors_sq_vs_rect_various_onefilter.py'):
    """test of various filterbanks with one orientation (but various #s of frequencies) on high density transformations set of squares versus rectangles
    Result: improves monotonically with number of frequencies, jumping a lot between one and two, but are, per training example, less good than two orientations."""

    D = v1_pull_protocol(depends_on)
    actualize(D)   
    
@protocolize()
def pull_gridded_gabors_sq_vs_rect_various_onefilter_mcc(depends_on = '../config/config_pull_gridded_gabors_sq_vs_rect_various_onefilter_mcc.py'):
    """
    """
    D = v1_pull_protocol(depends_on)
    actualize(D)  
    
@protocolize()
def pull_gridded_gabors_sq_vs_rect_various_onefilter_lrl(depends_on = '../config/config_pull_gridded_gabors_sq_vs_rect_various_onefilter_lrl.py'):
    """
    """
    D = v1_pull_protocol(depends_on)
    actualize(D)      
    
@protocolize()
def pull_gridded_gabors_sq_vs_rect_varioustwofilters(depends_on = '../config/config_pull_gridded_gabors_sq_vs_rect_varioustwofilters_evaluation.py'):
    """test of two-filter orthogonal orientation gridded gabor filterbanks of various kshapes and frequencies on high density transformations set of
    squares versus rectangles.
    RESULT: most do quite well.   Smaller kshape and higher frequencies in this test do better. """
    D = v1_pull_protocol(depends_on)
    actualize(D)      

@protocolize()
def pull_gridded_gabors_sq_vs_rect_various_two_orientation_mcc(depends_on = '../config/config_pull_gridded_gabors_sq_vs_rect_various_two_orientation_mcc.py'):
    """
    """
    D = v1_pull_protocol(depends_on)
    actualize(D)    
    
@protocolize()
def pull_gridded_gabors_sq_vs_rect_various_two_orientation_lrl(depends_on = '../config/config_pull_gridded_gabors_sq_vs_rect_various_two_orientation_lrl.py'):
    """
    """
    D = v1_pull_protocol(depends_on)
    actualize(D)        

@protocolize()
def pull_gridded_gabors_sq_vs_rect_various_twofrequency_filterbanks(depends_on = '../config/config_pull_gridded_gabors_sq_vs_rect_various_twofrequency_filterbanks.py'):
    """test of two-frequency gridded gabor filterbanks of various kshapes and fixed single orientation on high density transformations set of
    squares versus rectangles.
    RESULT: Varied """
    D = v1_pull_protocol(depends_on)
    actualize(D)      
   
###random gabors square vs. rect    
   
@protocolize()
def pull_random_gabors_sq_vs_rect_onefilter_screen(depends_on = '../config/config_pull_random_gabors_sq_vs_rect_onefilter_screen.py'):
    """screening 10 random one-filter gabors on high density transformations set of squares versus rectangles
    Result: all suck """
    D = v1_pull_protocol(depends_on)
    actualize(D)
    
    
@protocolize()
def pull_random_gabors_sq_vs_rect_twofilter_screen(depends_on = '../config/config_pull_random_gabors_sq_vs_rect_twofilter_screen.py'):
    """screening 10 randomly-oriented two-gabors filterbanks with fixed frequency and pahse on high density transformations set of squares versus rectangles
    Result: all suck""" 
    D = v1_pull_protocol(depends_on)
    actualize(D)    
    
    
@protocolize()
def pull_gabor_sq_vs_rect_twofilter_pump_training(depends_on = '../config/config_pull_gabor_sq_vs_rect_twofilter_pump_training.py'):
    """taking one of the best (but still bad) performing random two-filter gabors and pumping up traning examples on high density transformations 
    set of squares versus rectangles
    Result: still bad"""
    D = v1_pull_protocol(depends_on)
    actualize(D)    
    

###cairofilter activation tuning    
@protocolize()
def pull_cairofilters_sq_vs_rect_various_activations(depends_on = '../config/config_pull_cairofilters_sq_vs_rect_various_activations.py'):
    """tuning activation threshold on handcrafted cairo filters on high density transformations set of squares versus rectangles"""
    D = v1_pull_protocol(depends_on)
    actualize(D)  
    
@protocolize()
def pull_cairofilters_sq_vs_rect_various_activations_finetuning(depends_on = '../config/config_pull_cairofilters_sq_vs_rect_various_activations_finetuning.py'):
    """finetuning activation threshold on handcrafted cairo filters on high density transformations set of squares versus rectangles"""
    D = v1_pull_protocol(depends_on)
    actualize(D)      
    
@protocolize()
def pull_cairofilters_sq_vs_rect_various_activations_finefinetuning(depends_on = '../config/config_pull_cairofilters_sq_vs_rect_various_activations_finefinetuning.py'):
    """finefinetuning activation threshold on handcrafted cairo filters on high density transformations set of squares versus rectangles"""
    D = v1_pull_protocol(depends_on)
    actualize(D)   
    
@protocolize()
def pull_activation_tuned_cairofilters_sq_vs_rect(depends_on = '../config/config_pull_activation_tuned_cairofilters_sq_vs_rect.py'):
    """pumped-up trainining curve evaluation on  handcrafted cairo filters with optimized activation valued from finefinetuning on high density transformations set of squares versus rectangles
       result: you can pump up performance into > 95%. 
    """
    
    D = v1_pull_protocol(depends_on)
    actualize(D)          
    
    
###cairofilers
@protocolize()
def pull_cairofilters_sq_vs_rect_test2(depends_on = '../config/config_pull_cairofilters_sq_vs_rect_test2.py'):
    """
    Testing a slightly improved single-filter handcrafterd filterbank, related to original object.
    RESULT: Does better than original, with somewhat fewer training examples
    """
    D = v1_pull_protocol(depends_on)
    actualize(D)  
    
    
@protocolize()
def pull_cairofilters_sq_vs_rect_test3(depends_on = '../config/config_pull_cairofilters_sq_vs_rect_test3.py'):
    """
    Testing a slightly more improved single-filter handcrafterd filterbank, related to original object.
    RESULT: Does slightly even better, with somewhat fewer training examples
    """
    D = v1_pull_protocol(depends_on)
    actualize(D)    


###the orthogonality thing    
@protocolize()
def pull_gabors_sq_vs_rect_onespecialfilter(depends_on = '../config/config_pull_gabors_sq_vs_rect_onespecialfilter.py'):
    """
    Test to see if you could remove one of the two orthogonal copies of the best
    performing two-orthogonal-filter gabor (from greedy search). 
    
    RESULT: You can't.  That is, even though there's a good hyperplane defined
    with just the one filter, the SVM algorithm doesn't see it easily (you'd
    need a lot of test examples) but the second, orthogonal plane makes the
    separation much wider, so the SVM DOES see it.   At least with separating
    things like a square vs. a rectangle -- where there's a clear separation in
    edge space, there's a kind of "two filter orthogonality" principle.

    """
    D = v1_pull_protocol(depends_on)
    actualize(D)  
    
    
@protocolize()
def pull_cairofilters_sq_vs_rect_test4(depends_on = '../config/config_pull_cairofilters_sq_vs_rect_test4.py'):
    """
    Test using the two filter orthogonality principle with the original hand-crafted cairo-generated 
    filters that needed a lot of test examples to do well 
    
    RESULT: It works!  Just by adding a copy of the same filter, orthogonally in orientation space, 
    you get high performance with low numbers of training examples
        
    """
    D = v1_pull_protocol(depends_on)
    actualize(D)        

@protocolize()
def pull_cairofilters_sq_vs_rect_test5(depends_on = '../config/config_pull_cairofilters_sq_vs_rect_test5.py'):
    """
    Improvement on  two filter orthogonality principle with the better hand-crafted cairo-generated 
    filters using a primitive center-surround idea. 

    """
    D = v1_pull_protocol(depends_on)
    actualize(D)   

@protocolize()
def pull_cairofilters_sq_vs_rect_test6(depends_on = '../config/config_pull_cairofilters_sq_vs_rect_test6.py'):
    """
    Improvement on  two filter orthogonality principle with the better hand-crafted cairo-generated 
    filters using a slightly improved center-surround idea. 
 
    """
    D = v1_pull_protocol(depends_on)
    actualize(D)
    
    
###center surround    
@protocolize()
def pull_center_surround_sq_vs_rect(depends_on = '../config/config_pull_center_surround_sq_vs_rect.py'):
    """
    Using center surround construction procedure + orthogonal filter principle.
    RESULT:  it works pretty well.
    """
    D = v1_pull_protocol(depends_on)
    actualize(D)       
    
@protocolize()
def pull_center_surround_sq_vs_rect_test2(depends_on = '../config/config_pull_center_surround_sq_vs_rect_test2.py'):
    """
    same as other test but with different normin.kshape
    RESULT: it works about the same. 
    """
    D = v1_pull_protocol(depends_on)
    actualize(D)      
    

### transform-averaging the features before SVM    
@protocolize()
def pull_center_surround_sq_vs_rect_averaged(depends_on = '../config/config_pull_center_surround_sq_vs_rect_averaged.py'):
    """
    Implementing the idea that since the invariant hperplane should be itself invariant, you can just do this before the 
    SVM and get just as good results -- actually, better, since you can get esults with many fewer training examples
    RESULT: It works! You can reduce training example load quite a bit. 
    """
    D = v1_pull_protocol(depends_on)
    actualize(D)  

@protocolize()
def pull_center_surround_sq_vs_rect_averaged_test2(depends_on = '../config/config_pull_center_surround_sq_vs_rect_averaged_test2.py'):
    """
    same as other test, but different normin.kshape
    """
    D = v1_pull_protocol(depends_on)
    actualize(D)      
 

@protocolize()
def pull_center_surround_sq_vs_rect_averaged_nonorthogonalized(depends_on = '../config/config_pull_center_surround_sq_vs_rect_averaged_nonorthogonalized.py'):
    """
    center surround with averaging but not orthogonalization
    RESULT: bad
    """
    D = v1_pull_protocol(depends_on)
    actualize(D)      

@protocolize()
def pull_center_surround_sq_vs_rect_averaged_orthogonalized(depends_on = '../config/config_pull_center_surround_sq_vs_rect_averaged_orthogonalized.py'):
    """
    center surround with averaging and orthogonalization
    RESULT: great at small and large example sizes
    """
    D = v1_pull_protocol(depends_on)
    actualize(D)  
 
@protocolize()
def pull_center_surround_sq_vs_rect_nonaveraged_orthogonalized(depends_on = '../config/config_pull_center_surround_sq_vs_rect_nonaveraged_orthogonalized.py'):
    """
    center surround with averaging and orthogonalization
    RESULT: great at large-ish example sizes, but not so good small ones
    """
    D = v1_pull_protocol(depends_on)
    actualize(D) 
    
@protocolize()
def pull_center_surround_sq_vs_rect_nonaveraged_nonorthogonalized_svm_constants(depends_on = '../config/config_pull_center_surround_sq_vs_rect_nonaveraged_nonorthogonalized_svm_constants.py'):
    """
    trying out whether changing the SVM regularization constant makes fit better on a case where I can prove that the 
    hyperplane exists but SVM isn't finding it
    RESULT: changin the SVM constant doesn't do much
    """
    D = v1_pull_protocol(depends_on)
    actualize(D) 
    
@protocolize()
def pull_center_surround_sq_vs_rect_nonaveraged_nonorthogonalized_libsvm(depends_on = '../config/config_pull_center_surround_sq_vs_rect_nonaveraged_nonorthogonalized_libsvm.py'):
    """
    trying out whether a different SVM library  makes fit better on a case where I can prove that the 
    hyperplane exists but SVM isn't finding it
    RESULT: nope
    """
    D = v1_pull_protocol(depends_on)
    actualize(D)     

@protocolize()
def pull_center_surround_sq_vs_rect_nonaveraged_nonorthogonalized_mcc(depends_on = '../config/config_pull_center_surround_sq_vs_rect_nonaveraged_nonorthogonalized_mcc.py'):
    """
    trying out whether using a maximum correlation classifier (mcc) makes fit better on a case where I can prove that the 
    hyperplane exists but SVM isn't finding it
    RESULT: nope
    """
    D = v1_pull_protocol(depends_on)
    actualize(D)
    

@protocolize()
def pull_center_surround_sq_vs_rect_nonaveraged_nonorthogonalized_lrl(depends_on = '../config/config_pull_center_surround_sq_vs_rect_nonaveraged_nonorthogonalized_lrl.py'):
    """
    """
    D = v1_pull_protocol(depends_on)
    actualize(D) 

@protocolize()
def pull_center_surround_sq_vs_rect_nonaveraged_orthogonalized_mcc(depends_on = '../config/config_pull_center_surround_sq_vs_rect_nonaveraged_orthogonalized_mcc.py'):
    """
    """
    D = v1_pull_protocol(depends_on)
    actualize(D) 
    
@protocolize()
def pull_center_surround_sq_vs_rect_nonaveraged_orthogonalized_lrl(depends_on = '../config/config_pull_center_surround_sq_vs_rect_nonaveraged_orthogonalized_lrl.py'):
    """
    """
    D = v1_pull_protocol(depends_on)
    actualize(D)     


###center surround on circles vs squares
@protocolize()
def pull_center_surround_sq_vs_circle(depends_on = '../config/config_pull_center_surround_sq_vs_circle.py'):
    """
    testing whether the center surround construction works on circles vs. squares:
    RESULT: yes it does 
    """
    D = v1_pull_protocol(depends_on)
    actualize(D)   

@protocolize()
def pull_center_surround_sq_vs_circle_averaged(depends_on = '../config/config_pull_center_surround_sq_vs_circle_averaged.py'):
    """
    testing whether transform averaging in the center surround construction on circles vs. squares
    improves the performance for small numbers of training examples
    RESULT: yes it does     
    """
    D = v1_pull_protocol(depends_on)
    actualize(D)   


###gabors on circles vs squares

@protocolize()
def pull_gridded_gabors_sq_vs_circle_test(depends_on = '../config/config_pull_gridded_gabors_sq_vs_circle_test.py'):
    """
    circle vs square with translation on reduced-density dataset with very small objects with standard 96-gabor filterbank
    RESULT: The objects are so small and therefore low-resolutio that thre's not enogh data after input normalization and it totally fails. 
    """
    D = v1_pull_protocol(depends_on,)
    actualize(D) 
    

@protocolize()
def pull_gridded_gabors_sq_vs_circle_test2(depends_on = '../config/config_pull_gridded_gabors_sq_vs_circle_test2.py'):
    """
    attempt to rescue from previous test with larger objects.  
    RESULT: It works, you get 100% performance, e.g. standard 96-gabor filterbank works well. 
    """
    D = v1_pull_protocol(depends_on,)
    actualize(D)     
 

@protocolize()
def pull_gridded_gabors_sq_vs_circle(depends_on = '../config/config_pull_gridded_gabors_sq_vs_circle.py'):
    """
    "real" test with much higher translation-density dataset to separate circles from squares with standard 96-gabor filterbank 
    RESULT: 100% test accuracy
    """
    D = v1_pull_protocol(depends_on,)
    actualize(D)  
    

@protocolize()
def pull_gridded_gabors_sq_vs_circle_fewer_filters(depends_on = '../config/config_pull_gridded_gabors_sq_vs_circle_fewer_filters.py'):
    """
    reduced 48-gabor filter, otherwise same as previous
    RESULT: 100%
    """
    D = v1_pull_protocol(depends_on,)
    actualize(D)     
    
@protocolize()
def pull_gridded_gabors_sq_vs_circle_fewer_filters_2(depends_on = '../config/config_pull_gridded_gabors_sq_vs_circle_fewer_filters_2.py'):
    """
    reduced 12-gabor filter, otherwise same as previous
    RESULT: 100%
    
    """
    D = v1_pull_protocol(depends_on,)
    actualize(D)      
  
@protocolize()
def pull_gridded_gabors_sq_vs_circle_fewer_filters_3(depends_on = '../config/config_pull_gridded_gabors_sq_vs_circle_fewer_filters_3.py'):
    """
    reduced 2-gabor filter with one frequency and two orientations, otherwise same as previous
    RESULT: terrible.  
    """
    D = v1_pull_protocol(depends_on,)
    actualize(D)        
 
@protocolize()
def pull_gridded_gabors_sq_vs_circle_fewer_filters_4(depends_on = '../config/config_pull_gridded_gabors_sq_vs_circle_fewer_filters_4.py'):
    """
    4-gabor filter with two frequencies and two orientations
    RESULT: great.  

    """
    D = v1_pull_protocol(depends_on,)
    actualize(D)         

@protocolize()
def pull_gridded_gabors_sq_vs_circle_fewer_filters_5(depends_on = '../config/config_pull_gridded_gabors_sq_vs_circle_fewer_filters_5.py'):
    """
    same as pull_gridded_gabors_sq_vs_circle_fewer_filters_3, but with very different single frequency
    RESULT: just as terrible
    """
    D = v1_pull_protocol(depends_on,)
    actualize(D)         

@protocolize()
def pull_gridded_gabors_sq_vs_circle_fewer_filters_6(depends_on = '../config/config_pull_gridded_gabors_sq_vs_circle_fewer_filters_6.py'):
    """
    two-gabor filterbank with one orientation and two frequencies
    RESULT: Great
    """
    D = v1_pull_protocol(depends_on,)
    actualize(D)         

@protocolize()
def pull_gridded_gabors_sq_vs_circle_various_twofrequency_filterbanks(depends_on = '../config/config_pull_gridded_gabors_sq_vs_circle_various_twofrequency_filterbanks.py'):
    """
    exploring the two-frequency one-orientation space
    RESULTS: all pretty good, with some interesting trend.   SO upshot is that
    the way gabors solve cirlces vs squares, which after all are not cleanly separated in edge-space, 
    is to separate them in frequency space.  It's sort of a complement to the
    two-orthogonal filter priniple for the square-vs-rectangle (edge-separated) case. 
    """
    D = v1_pull_protocol(depends_on,)
    actualize(D)    
    
    
###gabors circle vs ellipse

@protocolize()
def pull_gridded_gabors_circle_vs_ellipse(depends_on = '../config/config_pull_gridded_gabors_circle_vs_ellipse.py'):
    """
    96 standard gabors on circles vs ellipse
    RESULT: Great
    """
    D = v1_pull_protocol(depends_on,)
    actualize(D)
    
@protocolize()
def pull_gridded_gabors_circle_vs_ellipse_fewer_filters(depends_on = '../config/config_pull_gridded_gabors_circle_vs_ellipse_fewer_filters.py'):
    """
    fewer gabors on circles vs ellipse
    RESULT: Great
    """
    D = v1_pull_protocol(depends_on,)
    actualize(D)
    
@protocolize()
def pull_gridded_gabors_circle_vs_ellipse_fewer_filters_2(depends_on = '../config/config_pull_gridded_gabors_circle_vs_ellipse_fewer_filters_2.py'):
    """
    fewer gabors on circles vs ellipse
    RESULT: Great
    """
    D = v1_pull_protocol(depends_on,)
    actualize(D)    
    
@protocolize()
def pull_gridded_gabors_circle_vs_ellipse_fewer_filters_3(depends_on = '../config/config_pull_gridded_gabors_circle_vs_ellipse_fewer_filters_3.py'):
    """
    fewer gabors on circles vs ellipse
    RESULT: Great
    """
    D = v1_pull_protocol(depends_on,)
    actualize(D)  
    
@protocolize()
def pull_gridded_gabors_circle_vs_ellipse_fewer_filters_4(depends_on = '../config/config_pull_gridded_gabors_circle_vs_ellipse_fewer_filters_4.py'):
    """
    two orients two frequencies
    RESULT: Great   
    """
    D = v1_pull_protocol(depends_on,)
    actualize(D)      
 
@protocolize()
def pull_gridded_gabors_circle_vs_ellipse_fewer_filters_5(depends_on = '../config/config_pull_gridded_gabors_circle_vs_ellipse_fewer_filters_5.py'):
    """
    two freqs one different orient
    RESULT: ok
    """
    D = v1_pull_protocol(depends_on,)
    actualize(D)  

@protocolize()
def pull_gridded_gabors_circle_vs_ellipse_fewer_filters_6(depends_on = '../config/config_pull_gridded_gabors_circle_vs_ellipse_fewer_filters_6.py'):
    """
    one orient one frequency
    RESULT: bad
    """
    D = v1_pull_protocol(depends_on,)
    actualize(D)
    
@protocolize()
def pull_gridded_gabors_circle_vs_ellipse_fewfilter_screening(depends_on = '../config/config_pull_gridded_gabors_circle_vs_ellipse_fewfilter_screening.py'):
    """
    various variety of one and two orientation and one and two-frequency on gabors circles vs ellipse
    RESULT: some work well, even with just one orientation -- but the two-orientation ones always do better due to orthogonalization
    """
    D = v1_pull_protocol(depends_on,)
    actualize(D)    
 
@protocolize()
def pull_center_surround_circle_vs_ellipse(depends_on = '../config/config_pull_center_surround_circle_vs_ellipse.py'):
    """
    center surround with circles vs ellipse, no orthogonalization -- using large v1 kernel (47 x 47) so that all the 
    whole image fits in the  filter 
    RESULT: OK (74% with 128 examples)
    """
    D = v1_pull_protocol(depends_on,)
    actualize(D)
    
@protocolize()
def pull_center_surround_circle_vs_ellipse_orthogonalized(depends_on = '../config/config_pull_center_surround_circle_vs_ellipse_orthogonalized.py'):
    """
    center surround with circles vs ellipse, with orthogonalization -- large kernel
    RESULT: OK      
    """
    D = v1_pull_protocol(depends_on,)
    actualize(D)  
    
@protocolize()
def pull_center_surround_circle_vs_ellipse_orthogonalized_2(depends_on = '../config/config_pull_center_surround_circle_vs_ellipse_orthogonalized.py'):
    """
    center surround with circles vs ellipse, with orthogonalization -- smaller kernel (32 x 32)
    RESULT: OK   
    """
    D = v1_pull_protocol(depends_on,)
    actualize(D)  
    
@protocolize()
def pull_center_surround_circle_vs_ellipse_orthogonalized_3(depends_on = '../config/config_pull_center_surround_circle_vs_ellipse_orthogonalized_3.py'):
    """
    center surround with circles vs ellipse, with orthogonalization -- smaller kernel (32 x 32) but somewhat compressed 
    RESULT: great at higher  numbers of training examples (%98 at 256) but not as good at smaller numbers
    """
    D = v1_pull_protocol(depends_on,)
    actualize(D)  
    
@protocolize()
def pull_center_surround_circle_vs_ellipse_averaged_orthogonalized_3(depends_on = '../config/config_pull_center_surround_circle_vs_ellipse_averaged_orthogonalized_3.py'):
    """
    same as orthogonalized_3 but using translation averaging
    RESULT: now it's good at smaller numbers of training examples
 
    """
    D = v1_pull_protocol(depends_on,)
    actualize(D)     
    
@protocolize()
def pull_center_surround_circle_vs_ellipse_orthogonalized_smaller(depends_on = '../config/config_pull_center_surround_circle_vs_ellipse_orthogonalized_smaller.py'):
    """

    """
    D = v1_pull_protocol(depends_on,)
    actualize(D) 
    
     
     
#=-=-=-=-=-=-=-=
#sq vs outline

#the basic gist of the results is that:
#	a) gridded gabors do this great with many filters
#   b) but not as great with few (e.g. two) filters
#   c) center surround with two filters doesn't do well either
#   d) and you need more examples
#   e) so it's a harder problem in for this v1 architecture and these filter bases ... certainly harder than it would be for pixels

@protocolize()
def pull_gridded_gabors_square_vs_outline(depends_on = '../config/config_pull_gridded_gabors_square_vs_outline.py'):
    """

    """
    D = v1_pull_protocol(depends_on,)
    actualize(D)    
@protocolize()
def pull_gridded_gabors_square_vs_outline_2(depends_on = '../config/config_pull_gridded_gabors_square_vs_outline_2.py'):
    """

    """
    D = v1_pull_protocol(depends_on,)
    actualize(D)        
@protocolize()
def pull_gridded_gabors_square_vs_outline_3(depends_on = '../config/config_pull_gridded_gabors_square_vs_outline_3.py'):
    """

    """
    D = v1_pull_protocol(depends_on,)
    actualize(D) 
@protocolize()
def pull_gridded_gabors_square_vs_outline_fewer_filters(depends_on = '../config/config_pull_gridded_gabors_square_vs_outline_fewer_filters.py'):
    """

    """
    D = v1_pull_protocol(depends_on,)
    actualize(D) 
    
@protocolize()
def pull_gridded_gabors_square_vs_outline_fewer_filters_2(depends_on = '../config/config_pull_gridded_gabors_square_vs_outline_fewer_filters_2.py'):
    """

    """
    D = v1_pull_protocol(depends_on,)
    actualize(D) 
@protocolize()
def pull_gridded_gabors_square_vs_outline_fewer_filters_3(depends_on = '../config/config_pull_gridded_gabors_square_vs_outline_fewer_filters_3.py'):
    """

    """
    D = v1_pull_protocol(depends_on,)
    actualize(D)  
@protocolize()
def pull_gridded_gabors_square_vs_outline_fewer_filters_4(depends_on = '../config/config_pull_gridded_gabors_square_vs_outline_fewer_filters_4.py'):
    """

    """
    D = v1_pull_protocol(depends_on,)
    actualize(D)

@protocolize()
def pull_gridded_gabors_square_vs_outline_fewer_filters_5(depends_on = '../config/config_pull_gridded_gabors_square_vs_outline_fewer_filters_5.py'):
    """

    """
    D = v1_pull_protocol(depends_on,)
    actualize(D)
    
@protocolize()
def pull_gridded_gabors_square_vs_outline_fewer_filters_6(depends_on = '../config/config_pull_gridded_gabors_square_vs_outline_fewer_filters_6.py'):
    """

    """
    D = v1_pull_protocol(depends_on,)
    actualize(D)  
@protocolize()
def pull_gridded_gabors_square_vs_outline_fewer_filters_7(depends_on = '../config/config_pull_gridded_gabors_square_vs_outline_fewer_filters_7.py'):
    """

    """
    D = v1_pull_protocol(depends_on,)
    actualize(D)     
@protocolize()
def pull_gridded_gabors_square_vs_outline_fewer_filters_8(depends_on = '../config/config_pull_gridded_gabors_square_vs_outline_fewer_filters_8.py'):
    """

    """
    D = v1_pull_protocol(depends_on,)
    actualize(D)         
          
@protocolize()
def pull_center_surround_square_vs_outline(depends_on = '../config/config_pull_center_surround_square_vs_outline.py'):
    """

    """
    D = v1_pull_protocol(depends_on,)
    actualize(D)
@protocolize()
def pull_center_surround_square_vs_outline_smaller(depends_on = '../config/config_pull_center_surround_square_vs_outline_smaller.py'):
    """

    """
    D = v1_pull_protocol(depends_on,)
    actualize(D)    
@protocolize()
def pull_center_surround_square_vs_outline_averaged(depends_on = '../config/config_pull_center_surround_square_vs_outline_averaged.py'):
    """

    """
    D = v1_pull_protocol(depends_on,)
    actualize(D)       
  


#=-=-=-=-=-=-=-=-=-=-= 
def v1_pull_evaluation_protocol(im_config_path,task_config_path,use_cpu = False,write=False):
    
    oplist = do_initialization(pull_initialize,args = (im_config_path,))    
    image_certificate = oplist[0]['outcertpaths'][0]
    model_certificate = oplist[1]['outcertpaths'][0]
    
    if use_cpu or not GPU_SUPPORT:    
        convolve_func = v1f.v1like_filter_numpy
    else:
        convolve_func = v1f.v1like_filter_pyfft

    config = get_config(task_config_path)
    task_config = config.pop('train_test')
    D = []
    for task in task_config:
        c = (config,task)       
        newhash = get_config_string(c)
        outfile = '../.performance_certificates/' + newhash
        op = ('svm_evaluation_' + newhash,train_test_pull,(outfile,task,image_certificate,model_certificate,convolve_func))
        D.append(op)

    if write:
        actualize(D)
    return D


def v1_pull_protocol(config_path,use_cpu = False,write=False):

    D = DBAdd(pull_initialize,args = (config_path,))
    
    oplist = do_initialization(pull_initialize,args = (config_path,))    
    image_certificate = oplist[0]['outcertpaths'][0]
    model_certificate = oplist[1]['outcertpaths'][0]
    
    if use_cpu or not GPU_SUPPORT:    
        convolve_func = v1f.v1like_filter_numpy
    else:
        convolve_func = v1f.v1like_filter_pyfft

    config = get_config(config_path)
    task_config = config.pop('train_test')
    
    for task in task_config:
        c = (config,task)       
        newhash = get_config_string(c)
        outfile = '../.performance_certificates/' + newhash
        op = ('svm_evaluation_' + newhash,train_test_pull,(outfile,task,image_certificate,model_certificate,convolve_func))
        D.append(op)

    if write:
        actualize(D)
    return D


def pull_initialize(config_path):
    config = get_config(config_path)    
    image_params = SON([('image',config['image'])])
    models_params = config['models']
    for model_params in models_params:
        if model_params['filter']['model_name'] in ['really_random','random_gabor']:
            #model_params['id'] = v1e.random_id()
            pass
    
    return [{'step':'generate_images','func':v1e.render_image, 'params':(image_params,)},                         
            {'step':'generate_models', 'func':v1e.get_filterbank,'params':(models_params,)},            
           ]

        
@activate(lambda x : (x[2],x[3]),lambda x : x[0])
def train_test_pull(outfile,task,image_certificate_file,model_certificate_file,convolve_func):

    conn = pm.Connection(document_class=bson.SON)
    db = conn['v1']
    
    perf_fs = gridfs.GridFS(db,'performance')
    
    model_coll = db['models.files']
    model_fs = gridfs.GridFS(db,'models')
    image_coll = db['raw_images.files']
    image_fs = gridfs.GridFS(db,'raw_images')
    
    image_certdict = cPickle.load(open(image_certificate_file))
    model_certdict = cPickle.load(open(model_certificate_file))
    print('using image certificate', image_certificate_file)
    print('using model certificate', model_certificate_file)    
    image_hash = image_certdict['run_hash']
    model_hash = model_certdict['run_hash']
    image_args = image_certdict['out_args']
    
    stats = ['test_accuracy','ap','auc','mean_ap','mean_auc','train_accuracy']    
    
    
    model_configs = get_most_recent_files(model_coll,{'__run_hash__':model_hash})
    
    if convolve_func == v1f.v1like_filter_pyfft:
        v1_pyfft.setup_pyfft()
    
    if isinstance(task,list):
        task_list = task
    else:
        task_list = [task]
    
    for task in task_list:
        classifier_kwargs = task.get('classifier_kwargs',{})    
        for m in model_configs:
            split_results = []
            splits = generate_splits(task,image_hash,'raw_images') 
            for (ind,split) in enumerate(splits):
                print ('split', ind)
                train_data = split['train_data']
                test_data = split['test_data']
                
                train_filenames = [t['filename'] for t in train_data]
                test_filenames = [t['filename'] for t in test_data]
                assert set(train_filenames).intersection(test_filenames) == set([])
                
                print('train feature extraction ...')
                train_features = sp.row_stack([transform_average(extract_features(im, image_fs, m, model_fs, convolve_func) , task.get('transform_average'),m) for im in train_data])
                print('test feature extraction ...')
                test_features = sp.row_stack([transform_average(extract_features(im, image_fs, m, model_fs, convolve_func) , task.get('transform_average'),m) for im in test_data])
                train_labels = split['train_labels']
                test_labels = split['test_labels']
                print('classifier ...') 
                res = svm.classify(train_features,train_labels,test_features,test_labels,classifier_kwargs)
     
                split_results.append(res)
        
            model_results = SON([])
            for stat in stats:
                if stat in split_results[0] and split_results[0][stat] != None:
                    model_results[stat] = sp.array([split_result[stat] for split_result in split_results]).mean()           
    
            out_record = SON([('model',m['config']['model']),
                           ('model_filename',m['filename']),
                           ('task',son_escape(task)),
                           ('images',son_escape(image_args)),
                           ('images_hash',image_hash),
                           ('models_hash',model_hash)
                         ])   
            filename = get_filename(out_record)
            out_record['filename'] = filename
            out_record.update(model_results)
            print('dump out ...') 
            out_data = cPickle.dumps(SON([('split_results',split_results),('splits',splits)]))
            
            perf_fs.put(out_data,**out_record)
 
    if convolve_func == v1f.v1like_filter_pyfft:
        v1_pyfft.cleanup_pyfft() 
      
    createCertificateDict(outfile,{'image_file':image_certificate_file,'models_file':model_certificate_file})
    
FEATURE_CACHE = {}

def get_from_cache(obj,cache):
    hash = hashlib.sha1(repr(obj)).hexdigest()
    if hash in cache:
        return cache[hash]
        
def put_in_cache(obj,value,cache):
    hash = hashlib.sha1(repr(obj)).hexdigest()
    cache[hash] = value

def extract_features(image_config, image_fs, model_config, model_fs,convolve_func):

    cached_val = get_from_cache((image_config,model_config),FEATURE_CACHE)
    if cached_val is not None:
        output = cached_val
    else:
        print('extracting', image_config, model_config)
        
        image_fh = image_fs.get_version(image_config['filename'])
        filter_fh = model_fs.get_version(model_config['filename'])
        
        m_config = model_config['config']['model']
        conv_mode = m_config['conv_mode']
        
        #preprocessing
        array = v1e.image2array(m_config ,image_fh)
        
        preprocessed,orig_imga = v1e.preprocess(array,m_config )
            
        #input normalization
        norm_in = v1e.norm(preprocessed,conv_mode,m_config.get('normin'))
        
        #filtering
        filtered = v1e.convolve(norm_in, filter_fh, m_config , convolve_func)
        
        #nonlinear activation
        activ = v1e.activate(filtered,m_config.get('activ'))
        
        #output normalization
        norm_out = v1e.norm(activ,conv_mode,m_config.get('normout'))
        #pooling
        pooled = v1e.pool(norm_out,conv_mode,m_config.get('pool'))
        
        if m_config.get('flatten',True):
            fvector_l = v1e.postprocess(norm_in,filtered,activ,norm_out,pooled,orig_imga,m_config.get('featsel'))
            output = sp.concatenate(fvector_l).ravel()
        else:
            output = pooled
        put_in_cache((image_config,model_config),output,FEATURE_CACHE)
    
    return output
    
import numpy as np
def transform_average(input,config,model_config):
    if config:
        averaged = []
        K = input.keys()
        K.sort()
        for cidx in K:
            averaged.append(average_transform(input[cidx],config,model_config))
        averaged = sp.concatenate(averaged)
        print(averaged)
        return averaged
    return input

def average_transform(input,config,M):
    if config['transform_name'] == 'translation':
        return input.sum(1).sum(0)
    elif config['transform_name'] == 'translation_and_orientation':
        model_config = M['config']['model'] 
        assert model_config.get('filter') and model_config['filter']['model_name'] == 'gridded_gabor'
        H = input.sum(1).sum(0) 
        norients = model_config['filter']['norients']
        phases = model_config['filter']['phases']
        nphases = len(phases)
        divfreqs = model_config['filter']['divfreqs']
        nfreqs = len(divfreqs)
        
        output = np.zeros((H.shape[0]/norients,)) 
        
        for freq_num in range(nfreqs):
            for phase_num in range(nphases):
                for orient_num in range(norients):
                    output[nphases*freq_num + phase_num] += H[norients*nphases*freq_num + nphases*orient_num + phase_num]
        
        return output
    elif config['transform_name'] == 'nothing':
        return input.ravel()
    elif config['transform_name'] == 'translation_and_fourier':
        return np.abs(np.fft.fft(input.sum(1).sum(0)))
    else:
        raise ValueError, 'Transform ' + str(config['transform_name']) + ' not recognized.'
    
def generate_splits(task_config,image_hash,colname):
    
    base_query = SON([('__run_hash__',image_hash)])
    ntrain = task_config['ntrain']
    ntest = task_config['ntest']
    ntrain_pos = task_config.get('ntrain_pos')
    N = task_config.get('N',10)
    query = task_config['query']  
    base_query.update(reach_in('config',task_config.get('universe',SON([]))))    
    cquery = reach_in('config',query)
    
    print ('query',cquery)
    print ('universe',base_query)
    
    return traintest.generate_split2('v1',colname,cquery,N,ntrain,ntest,ntrain_pos=ntrain_pos,universe=base_query)



#=-=-=-=-=-=-=-=-=-         

@protocolize()
def create_big_task_image_configs(depends_on = '../config/config_big_task.py'):
    config_path = depends_on
    image_protocol(config_path,write = True)
    

def image_protocol(config_path,write = False):

    config = get_config(config_path)

    image_colname = 'images_' + get_config_string(config['image'])
    image_certificate = '../.image_certificates/' + image_colname
    
    D = [('generate_image_configs',insert_random_cairo_configs,(image_certificate,image_colname,config['image']))]
    
    if write:
        actualize(D)
    return D
     
import rendering

@activate(lambda x : (), lambda x : x[0])    
def insert_random_cairo_configs(outfile,colname,config_gen):

    conn = pm.Connection(document_class = SON)
    db = conn['v1']
    
    X = rendering.cairo_random_config_gen(config_gen)
    
    db.drop_collection(colname + '.files')
    db.drop_collection(colname + '.chunks')
    
    fs = gridfs.GridFS(db,colname)
    
    for (i,x) in enumerate(X):
        if (i/100)*100 == i:
            print(i,x)
        x['image']['generator'] = 'cairo'    
        image_string = rendering.render_image(x['image']) 
        y = SON([('config',x)])
        filename = get_filename(x)
        y['filename'] = filename
        fs.put(image_string,**y)
        
    createCertificateDict(outfile,{'colname':colname,'args':config_gen})
     

@protocolize()
def gridded_gabors_big_task(depends_on = '../config/config_big_task_gridded_gabors.py'):
    """
    
    """
    config_path = depends_on
    v1_pull_random_images_protocol(config_path,write = True)
    
@protocolize()
def big_task_gridded_gabors_2(depends_on = '../config/config_big_task_gridded_gabors_2.py'):
    """
    
    """
    config_path = depends_on
    v1_pull_random_images_protocol(config_path,write = True)  
    
@protocolize()
def big_task_gridded_gabors_3(depends_on = '../config/config_big_task_gridded_gabors_3.py'):
    """
    
    """
    config_path = depends_on
    v1_pull_random_images_protocol(config_path,write = True)      

@protocolize()
def big_task_gridded_gabors_4(depends_on = '../config/config_big_task_gridded_gabors_4.py'):
    """
    
    """
    config_path = depends_on
    v1_pull_random_images_protocol(config_path,write = True)  

@protocolize()
def big_task_gridded_gabors_even_better(depends_on = '../config/config_big_task_gridded_gabors_even_better.py'):
    """
    
    """
    config_path = depends_on
    v1_pull_random_images_protocol(config_path,write = True)  

    
@protocolize()
def smaller_gridded_gabors_big_task(depends_on = '../config/config_big_task_smaller_gridded_gabors.py'):
    """
    
    """
    config_path = depends_on
    v1_pull_random_images_protocol(config_path,write = True)    

@protocolize()
def gridded_gabors_big_task_various_orients_freqs(depends_on = '../config/config_big_task_gridded_gabors_various_orients_freqs.py'):
    config_path = depends_on
    v1_pull_random_images_protocol(config_path,write = True)    

@protocolize()
def gridded_gabors_big_task_various_orients_freqs_2(depends_on = '../config/config_big_task_gridded_gabors_various_orients_freqs_2.py'):
    config_path = depends_on
    v1_pull_random_images_protocol(config_path,write = True)   

@protocolize()
def gridded_gabors_big_task_various_pump_training(depends_on = '../config/config_big_task_gridded_gabors_various_pump_training.py'):
    config_path = depends_on
    v1_pull_random_images_protocol(config_path,write = True)  
    
@protocolize()
def gridded_gabors_big_task_various_pump_training_2(depends_on = '../config/config_big_task_gridded_gabors_various_pump_training_2.py'):
    config_path = depends_on
    v1_pull_random_images_protocol(config_path,write = True)      


@protocolize()
def gridded_gabors_big_task_fewer_filters(depends_on = '../config/config_big_task_gridded_gabors_fewer_filters.py'):
    config_path = depends_on
    v1_pull_random_images_protocol(config_path,write = True)

@protocolize()
def gridded_gabors_big_task_fewer_filters_2(depends_on = '../config/config_big_task_gridded_gabors_fewer_filters_2.py'):
    config_path = depends_on
    v1_pull_random_images_protocol(config_path,write = True)   
    
@protocolize()
def gridded_gabors_big_task_fewer_filters_3(depends_on = '../config/config_big_task_gridded_gabors_fewer_filters_3.py'):
    config_path = depends_on
    v1_pull_random_images_protocol(config_path,write = True)       
    
@protocolize()
def gridded_gabors_big_task_fewer_filters_4(depends_on = '../config/config_big_task_gridded_gabors_fewer_filters_4.py'):
    config_path = depends_on
    v1_pull_random_images_protocol(config_path,write = True)  
    
@protocolize()
def gridded_gabors_big_task_fewer_filters_5(depends_on = '../config/config_big_task_gridded_gabors_fewer_filters_5.py'):
    config_path = depends_on
    v1_pull_random_images_protocol(config_path,write = True)  
    
@protocolize()
def gridded_gabors_big_task_fewer_filters_6(depends_on = '../config/config_big_task_gridded_gabors_fewer_filters_6.py'):
    config_path = depends_on
    v1_pull_random_images_protocol(config_path,write = True)  
    
@protocolize()
def gridded_gabors_big_task_fewer_filters_7(depends_on = '../config/config_big_task_gridded_gabors_fewer_filters_7.py'):
    config_path = depends_on
    v1_pull_random_images_protocol(config_path,write = True)  
    
@protocolize()
def gridded_gabors_big_task_more_filters(depends_on = '../config/config_big_task_gridded_gabors_more_filters.py'):
    config_path = depends_on
    v1_pull_random_images_protocol(config_path,write = True)  
     
@protocolize()
def gridded_gabors_big_task_better(depends_on = '../config/config_big_task_gridded_gabors_better.py'):
    config_path = depends_on
    v1_pull_random_images_protocol(config_path,write = True)  

@protocolize()
def gridded_gabors_big_task_orientation_averaging(depends_on = '../config/config_big_task_gridded_gabors_orientation_averaging.py'):
    config_path = depends_on
    v1_pull_random_images_protocol(config_path,write = True)  
    
@protocolize()
def gridded_gabors_big_task_orientation_averaging_2(depends_on = '../config/config_big_task_gridded_gabors_orientation_averaging_2.py'):
    config_path = depends_on
    v1_pull_random_images_protocol(config_path,write = True)      
    
 
@protocolize()
def gridded_gabors_big_task_orientation_averaging_3(depends_on = '../config/config_big_task_gridded_gabors_orientation_averaging_3.py'):
    config_path = depends_on
    v1_pull_random_images_protocol(config_path,write = True)       
    
@protocolize()
def gridded_gabors_big_task_orientation_averaging_4(depends_on = '../config/config_big_task_gridded_gabors_orientation_averaging_4.py'):
    config_path = depends_on
    v1_pull_random_images_protocol(config_path,write = True)       
        
@protocolize()
def gridded_gabors_big_task_orientation_averaging_5(depends_on = '../config/config_big_task_gridded_gabors_orientation_averaging_5.py'):
    config_path = depends_on
    v1_pull_random_images_protocol(config_path,write = True)   
    
@protocolize()
def gridded_gabors_big_task_orientation_averaging_6(depends_on = '../config/config_big_task_gridded_gabors_orientation_averaging_6.py'):
    config_path = depends_on
    v1_pull_random_images_protocol(config_path,write = True)       
                        
@protocolize()
def gridded_gabors_big_task_orientation_averaging_7(depends_on = '../config/config_big_task_gridded_gabors_orientation_averaging_7.py'):
    config_path = depends_on
    v1_pull_random_images_protocol(config_path,write = True)       
            
@protocolize()
def gridded_gabors_big_task_orientation_averaging_8(depends_on = '../config/config_big_task_gridded_gabors_orientation_averaging_8.py'):
    config_path = depends_on
    v1_pull_random_images_protocol(config_path,write = True)
    
@protocolize()
def gridded_gabors_big_task_orientation_averaging_9(depends_on = '../config/config_big_task_gridded_gabors_orientation_averaging_9.py'):
    config_path = depends_on
    v1_pull_random_images_protocol(config_path,write = True)    

@protocolize()
def gridded_gabors_big_task_orientation_averaging_10(depends_on = '../config/config_big_task_gridded_gabors_orientation_averaging_10.py'):
    config_path = depends_on
    v1_pull_random_images_protocol(config_path,write = True) 
    
@protocolize()
def gridded_gabors_big_task_orientation_averaging_11(depends_on = '../config/config_big_task_gridded_gabors_orientation_averaging_11.py'):
    config_path = depends_on
    v1_pull_random_images_protocol(config_path,write = True)     
    
@protocolize()
def gridded_gabors_big_task_orientation_averaging_12(depends_on = '../config/config_big_task_gridded_gabors_orientation_averaging_12.py'):
    config_path = depends_on
    v1_pull_random_images_protocol(config_path,write = True) 
    
@protocolize()
def gridded_gabors_big_task_orientation_averaging_13(depends_on = '../config/config_big_task_gridded_gabors_orientation_averaging_13.py'):
    config_path = depends_on
    v1_pull_random_images_protocol(config_path,write = True)  
    
    
@protocolize()
def gridded_gabors_big_task_orientation_averaging_14(depends_on = '../config/config_big_task_gridded_gabors_orientation_averaging_14.py'):
    config_path = depends_on
    v1_pull_random_images_protocol(config_path,write = True)          


@protocolize()
def gridded_gabors_big_task_orientation_averaging_15(depends_on = '../config/config_big_task_gridded_gabors_orientation_averaging_15.py'):
    config_path = depends_on
    v1_pull_random_images_protocol(config_path,write = True)       
    
@protocolize()
def gridded_gabors_big_task_orientation_averaging_16(depends_on = '../config/config_big_task_gridded_gabors_orientation_averaging_16.py'):
    config_path = depends_on
    v1_pull_random_images_protocol(config_path,write = True)           




@protocolize()
def gridded_gabors_sq_vs_rect_rot(depends_on = '../config/config_gridded_gabors_sq_vs_rect_rot.py'):
    """
    
    """
    config_path = depends_on
    v1_pull_random_images_protocol(config_path,write = True)  
    
    
@protocolize()
def gridded_gabors_sq_vs_rect_rot_2(depends_on = '../config/config_gridded_gabors_sq_vs_rect_rot_2.py'):
    config_path = depends_on
    v1_pull_random_images_protocol(config_path,write = True)   
    
    
@protocolize()
def gridded_gabors_sq_vs_rect_rot_3(depends_on = '../config/config_gridded_gabors_sq_vs_rect_rot_3.py'):
    config_path = depends_on
    v1_pull_random_images_protocol(config_path,write = True)       

@protocolize()
def gridded_gabors_sq_vs_rect_rot_4(depends_on = '../config/config_gridded_gabors_sq_vs_rect_rot_4.py'):
    config_path = depends_on
    v1_pull_random_images_protocol(config_path,write = True)      
    
@protocolize()
def gridded_gabors_sq_vs_rect_rot_5(depends_on = '../config/config_gridded_gabors_sq_vs_rect_rot_5.py'):
    config_path = depends_on
    v1_pull_random_images_protocol(config_path,write = True) 
    
@protocolize()
def gridded_gabors_sq_vs_rect_rot_6(depends_on = '../config/config_gridded_gabors_sq_vs_rect_rot_6.py'):
    config_path = depends_on
    v1_pull_random_images_protocol(config_path,write = True)     
          
          
@protocolize()
def gridded_gabors_sq_vs_rect_rot_7(depends_on = '../config/config_gridded_gabors_sq_vs_rect_rot_7.py'):
    config_path = depends_on
    v1_pull_random_images_protocol(config_path,write = True)  
          
    
@protocolize()
def gridded_gabors_sq_vs_rect_rot_8(depends_on = '../config/config_gridded_gabors_sq_vs_rect_rot_8.py'):
    config_path = depends_on
    v1_pull_random_images_protocol(config_path,write = True)      
    
@protocolize()
def gridded_gabors_sq_vs_rect_rot_9(depends_on = '../config/config_gridded_gabors_sq_vs_rect_rot_9.py'):
    config_path = depends_on
    v1_pull_random_images_protocol(config_path,write = True)     
    
@protocolize()
def gridded_gabors_sq_vs_rect_rot_10(depends_on = '../config/config_gridded_gabors_sq_vs_rect_rot_10.py'):
    config_path = depends_on
    v1_pull_random_images_protocol(config_path,write = True)         
          
                    

def v1_pull_random_images_protocol(config_path,use_cpu = False,write=False):

    D = DBAdd(model_initialize,args = (config_path,))
    oplist = do_initialization(model_initialize,args = (config_path,))    
    model_certificate = oplist[0]['outcertpaths'][0]
    
    im_prot = image_protocol(config_path,write = False)
    D += im_prot
    image_certificate = im_prot[0][2][0]
    
    if use_cpu or not GPU_SUPPORT:    
        convolve_func = v1f.v1like_filter_numpy
    else:
        convolve_func = v1f.v1like_filter_pyfft

    config = get_config(config_path)
    task_config = config.pop('train_test')
    
    for task in task_config:
        c = (config,task)       
        newhash = get_config_string(c)
        outfile = '../.performance_certificates/' + newhash
        op = ('svm_evaluation_' + newhash,train_test_pull_images,(outfile,task,image_certificate,model_certificate,convolve_func,config_path))
        D.append(op)

    if write:
        actualize(D)
    return D


@activate(lambda x : (x[2],x[3]),lambda x : x[0])
def train_test_pull_images(outfile,task,image_certificate_file,model_certificate_file,convolve_func,cpath):

    conn = pm.Connection(document_class=bson.SON)
    db = conn['v1']
    
    perf_fs = gridfs.GridFS(db,'performance')
    
    model_coll = db['models.files']
    model_fs = gridfs.GridFS(db,'models')
    
    print('using image certificate', image_certificate_file)
    print('using model certificate', model_certificate_file)   
    
    model_certdict = cPickle.load(open(model_certificate_file))
    model_hash = model_certdict['run_hash']
    
    image_certdict = cPickle.load(open(image_certificate_file))
    image_colname = image_certdict['colname']
    image_args = image_certdict['args']
    image_fs = gridfs.GridFS(db,image_colname)
    
    stats = ['test_accuracy','ap','auc','mean_ap','mean_auc','train_accuracy']    
    
    model_configs = get_most_recent_files(model_coll,{'__run_hash__':model_hash})
    
    if convolve_func == v1f.v1like_filter_pyfft:
        v1_pyfft.setup_pyfft()
    
    if isinstance(task,list):
        task_list = task
    else:
        task_list = [task]
    
    for task in task_list:
        print(task)
        classifier_kwargs = task.get('classifier_kwargs',{})    
        for m in model_configs:
            print('Evaluating model',m)
            split_results = []
            splits = generate_splits_2(task,image_colname) 
            for (ind,split) in enumerate(splits):
                print ('split', ind)
                train_data = split['train_data']
                test_data = split['test_data']
                
                train_filenames = [t['filename'] for t in train_data]
                test_filenames = [t['filename'] for t in test_data]
                assert set(train_filenames).intersection(test_filenames) == set([])
                
                print('train feature extraction ...')
                train_features = sp.row_stack([transform_average(extract_features(im, image_fs, m, model_fs, convolve_func,) , task.get('transform_average'),m) for im in train_data])
                print('test feature extraction ...')
                test_features = sp.row_stack([transform_average(extract_features(im, image_fs, m, model_fs, convolve_func) , task.get('transform_average'),m) for im in test_data])
                train_labels = split['train_labels']
                test_labels = split['test_labels']
    
                print('classifier ...')
                res = svm.classify(train_features,train_labels,test_features,test_labels,classifier_kwargs)
                print('Split test accuracy', res['test_accuracy'])
                split_results.append(res)
        
            model_results = SON([])
            for stat in stats:
                if stat in split_results[0] and split_results[0][stat] != None:
                    model_results[stat] = sp.array([split_result[stat] for split_result in split_results]).mean()           
    
            out_record = SON([('model',m['config']['model']),
                           ('model_filename',m['filename']),
                           ('task',son_escape(task)),
                           ('images',son_escape(image_args)),
                           ('image_colname',image_colname),
                           ('models_hash',model_hash)
                         ])   
            filename = get_filename(out_record)
            out_record['filename'] = filename
            out_record['config_path'] = cpath
            out_record.update(model_results)
            print('dump out ...')
            out_data = cPickle.dumps(SON([('split_results',split_results),('splits',splits)]))
            
            perf_fs.put(out_data,**out_record)
 
    if convolve_func == v1f.v1like_filter_pyfft:
        v1_pyfft.cleanup_pyfft() 
      
    createCertificateDict(outfile,{'image_file':image_certificate_file,'models_file':model_certificate_file})
    

def generate_splits_2(task_config,colname):
    
    base_query = SON([])
    ntrain = task_config['ntrain']
    ntest = task_config['ntest']
    ntrain_pos = task_config.get('ntrain_pos')
    ntest_pos = task_config.get('ntest_pos')
    N = task_config.get('N',10)
    query = task_config['query']  
    base_query.update(reach_in('config',task_config.get('universe',SON([]))))    
    cquery = reach_in('config',query)
    
    print('q',cquery)
    print('u',base_query)
 
    return traintest.generate_split2('v1',colname,cquery,N,ntrain,ntest,ntrain_pos=ntrain_pos,ntest_pos = ntest_pos,universe=base_query,use_negate = True)


def model_initialize(config_path):
    config = get_config(config_path)    
    image_params = SON([('image',config['image'])])
    models_params = config['models']

    return [{'step':'generate_models', 'func':v1e.get_filterbank,'params':(models_params,)}]     



#########
