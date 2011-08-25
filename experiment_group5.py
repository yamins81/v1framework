from starflow.protocols import Apply, Applies, protocolize

import deploy
import pythor_protocols as protocols




#################fourier#################
#I had the idea, perhaps like the v1 case of the polygon problem,
#doing fourier transform improve L1 performance in "real" (renderman) problems. 
#And if they could help L1s general, and if I could find a good structured solution 
#to higher levels (e.g. 3d gabors) that's at least _not worse_ than random,
#then the same ideas could maybe be applied there.   Very nifty.
#except it doesn't work. ion general fourier doesn't seem to add anything to performance
#although it doesn't degrade anything.
#
#So that is interesting because it suggests that somehow the rotation spectrum of the feature are truly interfering.
#So we really need to look at the features more closely.

@protocolize()
def ext_eval_ht_l1_gabor_reptile_vs_plant(depends_on=('../config/reptile_tasks.py',
                                                  '../config/ht_l1_gabor_top5_renderman_models.py',
                                                  '../config/ten_categories_images.py')):
    """
    seeing if fourier helps solve rotation problem (max transform)
    answer: no
    """
    a,b,c = depends_on
    protocols.extract_and_evaluate_protocol(a,b,c,convolve_func_name='numpy', write=True,parallel='semi')



@protocolize()
def ext_eval_ht_l1_gabor_reptile_vs_plant2(depends_on=('../config/reptile_tasks2.py',
                                                  '../config/ht_l1_gabor_top5_renderman_models.py',
                                                  '../config/ten_categories_images.py')):
    """
    seeing if fourier helps solve rotation problem with various_stats=True in transform_average
    answer: no           
    """
    a,b,c = depends_on
    protocols.extract_and_evaluate_protocol(a,b,c,convolve_func_name='numpy', write=True,parallel='semi')


@protocolize()
def ext_eval_various_l2_gabor_gabor_reptile_vs_plant(depends_on=('../config/reptile_tasks.py',
                                                  '../config/various_l2_gabor_gabor_models.py',
                                                  '../config/ten_categories_images.py')):
    """
    seeing if fourier helps solve rotation problem gridded 3d gabor solution 
    answer: no             
    """
    a,b,c = depends_on
    protocols.extract_and_evaluate_protocol(a,b,c,convolve_func_name='numpy', write=True,parallel='semi')



#################extraction for average vectors#################
#Basically, this does exaction of a bunch of sizes of L1s for one of the worst-offending pairwise problems 
#for rotation, e.g. planes vs. reptiles.  so we can do average-vector analysis. 
#Basically the answer (as one can see in nugget 3) is that under rotation the features are NOT being translated 
#with enough conserved, e.g. the general shape translates as expected but the magnitudes go up and down alot (for both planes and reptiles)
#so that the two clases switch magnitudes of features very sharply with rotation.   


@protocolize()
def make_various_l1_gabor_models(depends_on='../config/various_l1_gabor_models.py'):
    """
    """
    protocols.model_protocol(depends_on,parallel=False,write=True)    


@protocolize()
def extract_various_l1_gabors_reptile_and_planes(depends_on=('../config/reptile_plane_extraction.py',
                                                  '../config/various_l1_gabor_models.py',
                                                  '../config/ten_categories_images.py')):
    """
    hardest-to-separate pairwise task for rotation, to see what goes wrong with the feature traces.     
    result:  (see nugget 3) basically it looks like he general shape translates as expected but the magnitudes go up and down alot (for both planes and reptiles)
    so that the two clases switch magnitudes of features very sharply with rotation.   so maybe the gabor
    grid doesn't have fine enough orientation coverage?  (that doesn't "feel" right but we should check by double number of
    rotations)
    """
    a,b,c = depends_on
    protocols.extraction_protocol(a,b,c,convolve_func_name='numpy', write=True,parallel=True,save_to_db=True,batch_size=100)

@protocolize()
def extract_various_l1_gabors_reptile_and_planes_normed(depends_on=('../config/reptile_plane_extraction_normed.py',
                                                  '../config/various_l1_gabor_models.py',
                                                  '../config/ten_categories_images.py')):
    """

    """
    a,b,c = depends_on
    protocols.extraction_protocol(a,b,c,convolve_func_name='numpy', write=True,parallel=True,save_to_db=True,batch_size=100)


@protocolize()
def make_various_l1_gabor_models2(depends_on='../config/various_l1_gabor_models2.py'):
    """
    """
    protocols.model_protocol(depends_on,parallel=False,write=True) 

@protocolize()
def extract_various_l1_gabors2_reptile_and_planes(depends_on=('../config/reptile_plane_extraction.py',
                                                  '../config/various_l1_gabor_models2.py',
                                                  '../config/ten_categories_images.py')):
    """
    increasing the number of orientations ... does that smooth things out?         
    """
    a,b,c = depends_on
    protocols.extraction_protocol(a,b,c,convolve_func_name='numpy', write=True,parallel=True,save_to_db=True,batch_size=100)


@protocolize()
def extract_various_l1_gabors_face_and_table(depends_on=('../config/face_table_extraction.py',
                                                  '../config/various_l1_gabor_models.py',
                                                  '../config/ten_categories_images.py')):
    """
    two of the easier-to-separate (even with rotation) categories to see how the traces look different     
    """
    a,b,c = depends_on
    protocols.extraction_protocol(a,b,c,convolve_func_name='numpy', write=True,parallel=True,save_to_db=True,batch_size=100)


@protocolize()
def evaluate_various_l1_gabors_reptile_and_planes(depends_on=('../config/reptile_tasks3.py',
                                                  '../config/reptile_plane_extraction.py', 
                                                  '../config/various_l1_gabor_models.py',
                                                  '../config/ten_categories_images.py')):
    """
            
    """
    a,b,c,d = depends_on
    protocols.evaluate_protocol(a,b,c,d,convolve_func_name='numpy', write=True,parallel=False)
    
@protocolize()
def evaluate_various_l1_gabors_reptile_and_planes_normed(depends_on=('../config/reptile_tasks3.py',
                                                  '../config/reptile_plane_extraction_normed.py', 
                                                  '../config/various_l1_gabor_models.py',
                                                  '../config/ten_categories_images.py')):
    """
            
    """
    a,b,c,d = depends_on
    protocols.evaluate_protocol(a,b,c,d, write=True,parallel=False)

@protocolize()
def evaluate_various_l1_gabors2_reptile_and_planes(depends_on=('../config/reptile_tasks3.py',
                                                  '../config/reptile_plane_extraction.py', 
                                                  '../config/various_l1_gabor_models2.py',
                                                  '../config/ten_categories_images.py')):
    """
    adding more rotations doesn't seem to help at all (compare to evaluate_various_l1_gabors_reptile_and_planes)
    """
    a,b,c,d = depends_on
    protocols.evaluate_protocol(a,b,c,d,convolve_func_name='numpy', write=True,parallel=False)


@protocolize()
def evaluate_LRL_various_l1_gabors_reptile_and_planes(depends_on=('../config/reptile_tasks4.py',
                                                  '../config/reptile_plane_extraction.py', 
                                                  '../config/various_l1_gabor_models.py',
                                                  '../config/ten_categories_images.py')):
    """
            
    """
    a,b,c,d = depends_on
    protocols.evaluate_protocol(a,b,c,d,convolve_func_name='numpy', write=True,parallel=False)


@protocolize()
def make_various_l1_gabor_models_again(depends_on='../config/various_l1_gabor_models2.py'):
    """
    """
    protocols.model_protocol(depends_on,parallel=False,write=True) 

@protocolize()
def extract_various_l1_gabors_trans_and_inrot_percentile(depends_on=('../config/trans_inrot_extraction_percentile.py',
                                                  '../config/various_l1_gabor_models2.py',
                                                  '../config/ten_categories_images.py')):
    """

    """
    a,b,c = depends_on
    protocols.extraction_protocol(a,b,c,convolve_func_name='numpy', write=True,parallel=True,save_to_db=True,batch_size=100)


@protocolize()
def extract_top5_l1_gabor_renderman_percentile(depends_on=('../config/renderman_extraction_percentile.py',
                                                  '../config/ht_l1_gabor_top5_renderman_models.py',
                                                  '../config/ten_categories_images.py')):
    """

    """
    a,b,c = depends_on
    protocols.extraction_protocol(a,b,c,convolve_func_name='numpy', write=True,parallel=True,save_to_db=True,batch_size=200)


@protocolize()
def evaluate_top5_l1_gabor_renderman_percentile(depends_on=('../config/renderman_percentile_tasks.py',
                                                  '../config/renderman_extraction_percentile.py', 
                                                  '../config/ht_l1_gabor_top5_renderman_models.py',
                                                  '../config/ten_categories_images.py')):
    """
            
    """
    a,b,c,d = depends_on
    protocols.evaluate_protocol(a,b,c,d,c write=True,parallel = False, use_db = True)


@protocolize()
def make_top5_manylevel_renderman_models(depends_on='../config/ht_manylevel_top5_renderman_models.py'):
    """
    """
    protocols.model_protocol(depends_on,parallel=False,write=True) 

@protocolize()
def extract_top5_manylevel_renderman_percentile(depends_on=('../config/renderman_extraction_percentile.py',
                                                  '../config/ht_manylevel_top5_renderman_models.py',
                                                  '../config/ten_categories_images.py')):
    """

    """
    a,b,c = depends_on
    protocols.extraction_protocol(a,b,c,convolve_func_name='numpy', write=True,parallel=True,save_to_db=True,batch_size=100)




#######trying to simplify even further
@Applies(deploy.images,args=('../config/reptiles_and_planes_images.py',True))
def generate_reptiles_and_planes_images():
    Apply()
    
@protocolize()
def extract_various_l1_gabors_reptiles_and_planes_subtasks(depends_on=('../config/reptile_plane_extraction.py',
                                                  '../config/various_l1_gabor_models.py',
                                                  '../config/reptiles_and_planes_images.py')):
    """
    """
    a,b,c = depends_on
    protocols.extraction_protocol(a,b,c,convolve_func_name='numpy', write=True,parallel=True,save_to_db=True,batch_size=100)

@protocolize()
def extract_various_l1_gabors_reptiles_and_planes_subtasks_percentile(depends_on=('../config/reptile_plane_extraction_percentile.py',
                                                  '../config/various_l1_gabor_models.py',
                                                  '../config/reptiles_and_planes_images.py')):
    """
    """
    a,b,c = depends_on
    protocols.extraction_protocol(a,b,c,convolve_func_name='numpy', write=True,parallel=True,save_to_db=True,batch_size=100)


@protocolize()
def evaluate_various_l1_gabors_reptiles_and_planes_subtasks(depends_on=('../config/reptile_subtasks.py',
                                                  '../config/reptile_plane_extraction.py', 
                                                  '../config/various_l1_gabor_models.py',
                                                  '../config/reptiles_and_planes_images.py')):
    """
            
    """
    a,b,c,d = depends_on
    protocols.evaluate_protocol(a,b,c,d,write=True,parallel=True)
    
    
@protocolize()
def evaluate_various_l1_gabors_reptiles_and_planes_subtasks2(depends_on=('../config/reptile_subtasks2.py',
                                                  '../config/reptile_plane_extraction.py', 
                                                  '../config/various_l1_gabor_models.py',
                                                  '../config/reptiles_and_planes_images.py')):
    """
     geck & turtle more examples       
    """
    a,b,c,d = depends_on
    protocols.evaluate_protocol(a,b,c,d,write=True,parallel=False)

@protocolize()
def evaluate_various_l1_gabors_reptiles_and_planes_subtasks3(depends_on=('../config/reptile_subtasks3.py',
                                                  '../config/reptile_plane_extraction.py', 
                                                  '../config/various_l1_gabor_models.py',
                                                  '../config/reptiles_and_planes_images.py')):
    """
     geck & turtle vs. specific planes more examples       
    """
    a,b,c,d = depends_on
    protocols.evaluate_protocol(a,b,c,d,write=True,parallel=False)

@protocolize()
def evaluate_various_l1_gabors_reptiles_and_planes_subtasks4(depends_on=('../config/reptile_subtasks4.py',
                                                  '../config/reptile_plane_extraction.py', 
                                                  '../config/various_l1_gabor_models.py',
                                                  '../config/reptiles_and_planes_images.py')):
    """
     leatherback & monitor lizard (!) vs. specific planes more examples 
     result:  basically the phenomenon is less obvious
    """
    a,b,c,d = depends_on
    protocols.evaluate_protocol(a,b,c,d,write=True,parallel=False)
    

@protocolize()
def evaluate_various_l1_gabors_reptiles_and_planes_subtasks5(depends_on=('../config/reptile_subtasks5.py',
                                                  '../config/reptile_plane_extraction.py', 
                                                  '../config/various_l1_gabor_models.py',
                                                  '../config/reptiles_and_planes_images.py')):
    """
     monitor lizard vs pairs of planes ... to see if we can make the T-I phenomenon appear greater   
    """
    a,b,c,d = depends_on
    protocols.evaluate_protocol(a,b,c,d,write=True,parallel=True)
    
@protocolize()
def evaluate_various_l1_gabors_reptiles_and_planes_subtasks6(depends_on=('../config/reptile_subtasks6.py',
                                                  '../config/reptile_plane_extraction.py', 
                                                  '../config/various_l1_gabor_models.py',
                                                  '../config/reptiles_and_planes_images.py')):
    """
     monitor lizard (!) vs all plans with more examples     
    """
    a,b,c,d = depends_on
    protocols.evaluate_protocol(a,b,c,d,write=True,parallel=False)
    
    
@protocolize()
def evaluate_various_l1_gabors_reptiles_and_planes_subtasks7(depends_on=('../config/reptile_subtasks7.py',
                                                  '../config/reptile_plane_extraction.py', 
                                                  '../config/various_l1_gabor_models.py',
                                                  '../config/reptiles_and_planes_images.py')):
    """
     reptiles vs planes with more examples     
    """
    a,b,c,d = depends_on
    protocols.evaluate_protocol(a,b,c,d,write=True,parallel=False)    
    
    
    
@protocolize()
def evaluate_various_l1_gabors_lizard_vs_plane_percentile(depends_on=('../config/reptile_subtasks4.py',
                                                  '../config/reptile_plane_extraction_percentile.py', 
                                                  '../config/various_l1_gabor_models.py',
                                                  '../config/reptiles_and_planes_images.py')):
    """
     leatherback & monitor lizard (!) vs. specific planes with percentile
    """
    a,b,c,d = depends_on
    protocols.evaluate_protocol(a,b,c,d,write=True,parallel=True)

@protocolize()
def evaluate_various_l1_gabors_lizard_vs_particular_plane_percentile(depends_on=('../config/reptile_subtasks8.py',
                                                  '../config/reptile_plane_extraction_percentile.py', 
                                                  '../config/various_l1_gabor_models.py',
                                                  '../config/reptiles_and_planes_images.py')):
    """
     leatherback & monitor lizard (!) vs. specific planes with percentile
    """
    a,b,c,d = depends_on
    protocols.evaluate_protocol(a,b,c,d,write=True,parallel=True)
    
@protocolize()
def evaluate_various_l1_gabors_reptiles_and_planes_percentile(depends_on=('../config/reptile_subtasks7.py',
                                                  '../config/reptile_plane_extraction_percentile.py', 
                                                  '../config/various_l1_gabor_models.py',
                                                  '../config/reptiles_and_planes_images.py')):
    """
     reptiles vs planes with percentile  
    """
    a,b,c,d = depends_on
    protocols.evaluate_protocol(a,b,c,d,write=True,parallel=True)
    
@protocolize()
def evaluate_various_l1_gabors_reptiles_and_planes_percentile_more(depends_on=('../config/reptile_subtasks9.py',
                                                  '../config/reptile_plane_extraction_percentile.py', 
                                                  '../config/various_l1_gabor_models.py',
                                                  '../config/reptiles_and_planes_images.py')):
    """
     reptiles vs planes with percentile   and more exmaples etc .... 
    """
    a,b,c,d = depends_on
    protocols.evaluate_protocol(a,b,c,d,write=True,parallel=True)
