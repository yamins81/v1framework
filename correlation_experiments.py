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
