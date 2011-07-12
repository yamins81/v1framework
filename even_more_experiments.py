from starflow.protocols import Apply, Applies, protocolize

import deploy
import pythor_protocols as protocols



@protocolize()
def make_ht_l2_gabor_random_squared_models(depends_on='../config/ht_l2_gabor_random_squared_models.py'):
    """
  
    """
    protocols.model_protocol(depends_on,parallel=False,write=True)




@protocolize()
def ext_eval_ht_l2_random_random_larger_renderman(depends_on=('../config/renderman_tasks_for_ht2.py',
                                                  '../config/ht_l2_gabor_random_squared_models.py',
                                                  '../config/ten_categories_images.py')):
    """
    """
    protocols.extract_and_evaluate_protocol('../config/renderman_tasks_for_ht2.py',
                                            '../config/ht_l2_gabor_random_squared_models.py',
                                            '../config/ten_categories_images.py',
                                            convolve_func_name='numpy', write=True,parallel=True)

