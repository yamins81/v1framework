from starflow.protocols import Apply, Applies, protocolize

import deploy
import pythor_protocols as protocols


                                            
@protocolize()
def make_best_l3_model(depends_on='../config/best_l3_model.py'):
    protocols.model_protocol(depends_on,parallel=False,write=True)

                                            
@protocolize()
def ext_eval_best_l3_exploratory_renderman(depends_on=('../config/exploratory_renderman_tasks.py',
                                                  '../config/best_l3_model.py',
                                                  '../config/ten_categories_images.py')):
    protocols.extract_and_evaluate_protocol('../config/exploratory_renderman_tasks.py',
                                            '../config/best_l3_model.py',
                                            '../config/ten_categories_images.py',
                                            convolve_func_name='numpy', write=True,parallel=False)
                                            
