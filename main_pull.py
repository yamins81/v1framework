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
    """test of 1-filter 'gridded' gabor filterbank on high density transformations set of squares versus rectangles
    Result: not great""" 
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

    """
    D = v1_pull_protocol(depends_on)
    actualize(D) 
    
@protocolize()
def pull_center_surround_sq_vs_rect_nonaveraged_nonorthogonalized_libsvm(depends_on = '../config/config_pull_center_surround_sq_vs_rect_nonaveraged_nonorthogonalized_libsvm.py'):
    """

    """
    D = v1_pull_protocol(depends_on)
    actualize(D)     


###center surround on circles vs squares
@protocolize()
def pull_center_surround_sq_vs_circle(depends_on = '../config/config_pull_center_surround_sq_vs_circle.py'):
    """
    
    """
    D = v1_pull_protocol(depends_on)
    actualize(D)   

@protocolize()
def pull_center_surround_sq_vs_circle_averaged(depends_on = '../config/config_pull_center_surround_sq_vs_circle_averaged.py'):
    """
    
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

    """
    D = v1_pull_protocol(depends_on,)
    actualize(D)
    
@protocolize()
def pull_gridded_gabors_circle_vs_ellipse_fewer_filters(depends_on = '../config/config_pull_gridded_gabors_circle_vs_ellipse_fewer_filters.py'):
    """

    """
    D = v1_pull_protocol(depends_on,)
    actualize(D)
    
@protocolize()
def pull_gridded_gabors_circle_vs_ellipse_fewer_filters_2(depends_on = '../config/config_pull_gridded_gabors_circle_vs_ellipse_fewer_filters_2.py'):
    """

    """
    D = v1_pull_protocol(depends_on,)
    actualize(D)    
    
@protocolize()
def pull_gridded_gabors_circle_vs_ellipse_fewer_filters_3(depends_on = '../config/config_pull_gridded_gabors_circle_vs_ellipse_fewer_filters_3.py'):
    """

    """
    D = v1_pull_protocol(depends_on,)
    actualize(D)  
    
@protocolize()
def pull_gridded_gabors_circle_vs_ellipse_fewer_filters_4(depends_on = '../config/config_pull_gridded_gabors_circle_vs_ellipse_fewer_filters_4.py'):
    """

    """
    D = v1_pull_protocol(depends_on,)
    actualize(D)      
 
@protocolize()
def pull_gridded_gabors_circle_vs_ellipse_fewer_filters_5(depends_on = '../config/config_pull_gridded_gabors_circle_vs_ellipse_fewer_filters_5.py'):
    """

    """
    D = v1_pull_protocol(depends_on,)
    actualize(D)  

@protocolize()
def pull_gridded_gabors_circle_vs_ellipse_fewer_filters_6(depends_on = '../config/config_pull_gridded_gabors_circle_vs_ellipse_fewer_filters_6.py'):
    """

    """
    D = v1_pull_protocol(depends_on,)
    actualize(D)
    
@protocolize()
def pull_gridded_gabors_circle_vs_ellipse_fewfilter_screening(depends_on = '../config/config_pull_gridded_gabors_circle_vs_ellipse_fewfilter_screening.py'):
    """

    """
    D = v1_pull_protocol(depends_on,)
    actualize(D)    
 
@protocolize()
def pull_center_surround_circle_vs_ellipse(depends_on = '../config/config_pull_center_surround_circle_vs_ellipse.py'):
    """

    """
    D = v1_pull_protocol(depends_on,)
    actualize(D)
    
@protocolize()
def pull_center_surround_circle_vs_ellipse_orthogonalized(depends_on = '../config/config_pull_center_surround_circle_vs_ellipse_orthogonalized.py'):
    """

    """
    D = v1_pull_protocol(depends_on,)
    actualize(D)  
    
@protocolize()
def pull_center_surround_circle_vs_ellipse_orthogonalized_2(depends_on = '../config/config_pull_center_surround_circle_vs_ellipse_orthogonalized.py'):
    """

    """
    D = v1_pull_protocol(depends_on,)
    actualize(D)  
    
@protocolize()
def pull_center_surround_circle_vs_ellipse_orthogonalized_3(depends_on = '../config/config_pull_center_surround_circle_vs_ellipse_orthogonalized_3.py'):
    """

    """
    D = v1_pull_protocol(depends_on,)
    actualize(D)  
    
@protocolize()
def pull_center_surround_circle_vs_ellipse_averaged_orthogonalized_3(depends_on = '../config/config_pull_center_surround_circle_vs_ellipse_averaged_orthogonalized_3.py'):
    """

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
			splits = generate_splits(task,image_hash) 
			for (ind,split) in enumerate(splits):
				print ('split', ind)
				train_data = split['train_data']
				test_data = split['test_data']
				
				train_filenames = [t['filename'] for t in train_data]
				test_filenames = [t['filename'] for t in test_data]
				assert set(train_filenames).intersection(test_filenames) == set([])
				
				train_features = sp.row_stack([transform_average(extract_features(im, image_fs, m, model_fs, convolve_func) , task.get('transform_average')) for im in train_data])
				test_features = sp.row_stack([transform_average(extract_features(im, image_fs, m, model_fs, convolve_func) , task.get('transform_average')) for im in test_data])
				train_labels = split['train_labels']
				test_labels = split['test_labels']
	
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
    

def transform_average(input,config):
    if config:
        averaged = []
        K = input.keys()
        K.sort()
        for cidx in K:
            averaged.append(average_transform(input[cidx],config))
        averaged = sp.concatenate(averaged)
        print(averaged)
        return averaged
    return input

def average_transform(input,config):
    if config['transform_name'] == 'translation':
        return input.sum(1).sum(0)
    else:
        raise ValueError, 'Transform ' + str(config['transform_name']) + ' not recognized.'
    
def generate_splits(task_config,image_hash):
    
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
    
    return [traintest.generate_split2('v1','raw_images',cquery,ntrain,ntest,ntrain_pos=ntrain_pos,universe=base_query) for ind in range(N)]
        