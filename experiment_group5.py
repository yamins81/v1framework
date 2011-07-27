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

