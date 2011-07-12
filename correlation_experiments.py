from starflow.protocols import Apply, Applies, protocolize

import pythor_protocols as protocols


@protocolize()
def make_various_l1_gabor_models(depends_on='../config/various_l1_gabors.py'):
    """

    """
    protocols.model_protocol(depends_on,parallel=False,write=True)


@protocolize()
def correlation_l1_gabor_polygon(depends_on=('../config/polygon_correlation_tasks.py',
                                                  '../config/various_l1_gabors.py',
                                                  '../config/polygon_task.py')):
    protocols.get_corr_protocol('../config/polygon_correlation_tasks.py',
                                '../config/various_l1_gabors.py',
                                '../config/polygon_task.py',
                                convolve_func_name='numpy', write=True,parallel=True)


@protocolize()
def correlation_l1_gabor_renderman(depends_on=('../config/renderman_correlation_tasks.py',
                                                  '../config/various_l1_gabors.py',
                                                  '../config/ten_categories_images.py')):
    protocols.get_corr_protocol('../config/renderman_correlation_tasks.py',
                                '../config/various_l1_gabors.py',
                                '../config/ten_categories_images.py',
                                convolve_func_name='numpy', write=True,parallel=True)


@protocolize()
def make_ht_l1_gabor_models_for_corr(depends_on='../config/ht_l1_gabor_models_for_corr.py'):
    """

    """
    protocols.model_protocol(depends_on,parallel=False,write=True)



@protocolize()
def correlation_ht_l1_gabor_renderman(depends_on=('../config/renderman_correlation_tasks2.py',
                                                  '../config/ht_l1_gabor_models_for_corr.py',
                                                  '../config/ten_categories_images.py')):
    protocols.get_corr_protocol('../config/renderman_correlation_tasks2.py',
                                '../config/ht_l1_gabor_models_for_corr.py',
                                '../config/ten_categories_images.py',
                                convolve_func_name='numpy', write=True,parallel=True)


@protocolize()
def make_corr_test_3_model(depends_on='../config/l2_gabor_corr_test_model3.py'):
    """

    """
    protocols.model_protocol(depends_on,parallel=False,write=True)



@protocolize()
def correlation_test_3_model(depends_on=('../config/renderman_correlation_tasks2.py',
                                                  '../config/l2_gabor_corr_test_model3.py',
                                                  '../config/ten_categories_images.py')):
    protocols.get_corr_protocol('../config/renderman_correlation_tasks2.py',
                                '../config/l2_gabor_corr_test_model3.py',
                                '../config/ten_categories_images.py',
                                convolve_func_name='numpy', write=True,parallel=True)
                                
                                
@protocolize()
def make_ht_l2_gabor_corr_subset_models(depends_on='../config/ht_l2_gabor_corr_subset_models.py'):
    """
    """
    protocols.model_protocol(depends_on,parallel=False,write=True)


@protocolize()
def ext_eval_ht_l2_gabor_corr_subset_renderman(depends_on=('../config/renderman_tasks_for_ht2.py',
                                                  '../config/ht_l2_gabor_corr_subset_models.py',
                                                  '../config/ten_categories_images.py')):
    """
    basically a failure e.g. no better than random (if anything, slightly worse)
    overall: 43.60, 29.20, 36.33, 38.00, 2.65
    256 filters: 42.80, 29.20, 35.86, 37.73, 2.72
    384: 43.60, 31.20, 36.79, 38.27, 2.50
    """
    protocols.extract_and_evaluate_protocol('../config/renderman_tasks_for_ht2.py',
                                            '../config/ht_l2_gabor_corr_subset_models.py',
                                            '../config/ten_categories_images.py',
                                            convolve_func_name='numpy', write=True,parallel=True)
                                