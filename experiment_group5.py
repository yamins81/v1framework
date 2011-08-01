from starflow.protocols import Apply, Applies, protocolize

import deploy
import pythor_protocols as protocols




@protocolize()
def ext_eval_ht_l1_gabor_reptile_vs_plant(depends_on=('../config/reptile_tasks.py',
                                                  '../config/ht_l1_gabor_top5_renderman_models.py',
                                                  '../config/ten_categories_images.py')):
    """
            
    """
    a,b,c = depends_on
    protocols.extract_and_evaluate_protocol(a,b,c,convolve_func_name='numpy', write=True,parallel='semi')



@protocolize()
def ext_eval_ht_l1_gabor_reptile_vs_plant2(depends_on=('../config/reptile_tasks2.py',
                                                  '../config/ht_l1_gabor_top5_renderman_models.py',
                                                  '../config/ten_categories_images.py')):
    """
            
    """
    a,b,c = depends_on
    protocols.extract_and_evaluate_protocol(a,b,c,convolve_func_name='numpy', write=True,parallel='semi')


@protocolize()
def ext_eval_various_l2_gabor_gabor_reptile_vs_plant(depends_on=('../config/reptile_tasks.py',
                                                  '../config/various_l2_gabor_gabor_models.py',
                                                  '../config/ten_categories_images.py')):
    """
            
    """
    a,b,c = depends_on
    protocols.extract_and_evaluate_protocol(a,b,c,convolve_func_name='numpy', write=True,parallel='semi')



#################extraction for average vectors


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
