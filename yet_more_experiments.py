from starflow.protocols import Apply, Applies, protocolize

import deploy
import pythor_protocols as protocols


@Applies(deploy.images,args=('../config/faces_images.py',True))
def generate_faces_images():
    Apply()

@protocolize()
def ext_eval_various_random_l3_face_subtasks(depends_on=('../config/face_subtasks.py',
                                                  '../config/various_random_l3_grayscale_models.py',
                                                  '../config/faces_images.py')):
    """
    all: (21.269841269841269, 13.333333333333334, 16.884920634920636, 2.6135554487418613)
    just trans: (100.0, 98.412698412698418, 99.642857142857153, 0.50037778622692353)
    in-plane: (96.19047619047619, 82.222222222222214, 90.813492063492049, 3.8086453186818607)
    scale; (98.412698412698418, 89.523809523809518, 95.416666666666657, 2.4045621741142726)
    inplane and scale: (52.04081632653061, 39.455782312925173, 46.301020408163268, 3.5853208104055021)
    """
    protocols.extract_and_evaluate_protocol('../config/face_subtasks.py',
                                            '../config/various_random_l3_grayscale_models.py',
                                            '../config/faces_images.py',
                                            convolve_func_name='numpy', write=True,parallel=True)



###############


@protocolize()
def ext_eval_ht_l2_random_random_renderman(depends_on=('../config/renderman_tasks_for_ht.py',
                                                  '../config/ht_l2_random_random_models.py',
                                                  '../config/ten_categories_images.py')):
    """
    
    """
    protocols.extract_and_evaluate_protocol('../config/renderman_tasks_for_ht.py',
                                            '../config/ht_l2_random_random_models.py',
                                            '../config/ten_categories_images.py',
                                            convolve_func_name='numpy', write=True,parallel=True)

@protocolize()
def ext_eval_ht_l2_gabor_random_renderman(depends_on=('../config/renderman_tasks_for_ht.py',
                                                  '../config/ht_l2_gabor_random_models.py',
                                                  '../config/ten_categories_images.py')):
    """
    
    """
    protocols.extract_and_evaluate_protocol('../config/renderman_tasks_for_ht.py',
                                            '../config/ht_l2_gabor_random_models.py',
                                            '../config/ten_categories_images.py',
                                            convolve_func_name='numpy', write=True,parallel=True)


@protocolize()
def ext_eval_ht_l2_freq_renderman(depends_on=('../config/renderman_tasks_for_ht.py',
                                                  '../config/ht_l2_freq_models.py',
                                                  '../config/ten_categories_images.py')):
    """
    
    """
    protocols.extract_and_evaluate_protocol('../config/renderman_tasks_for_ht.py',
                                            '../config/ht_l2_freq_models.py',
                                            '../config/ten_categories_images.py',
                                            convolve_func_name='numpy', write=True,parallel=True)


@protocolize()
def make_ht_l1_gabor_models(depends_on='../config/ht_l1_gabor_models_more.py'):
    """
  
    """
    protocols.model_protocol(depends_on,parallel=False,write=True)


@protocolize()
def ext_eval_ht_l1_gabor_renderman(depends_on=('../config/renderman_tasks_for_ht2.py',
                                                  '../config/ht_l1_gabor_models_more.py',
                                                  '../config/ten_categories_images.py')):
    """
    
    """
    protocols.extract_and_evaluate_protocol('../config/renderman_tasks_for_ht2.py',
                                            '../config/ht_l1_gabor_models_more.py',
                                            '../config/ten_categories_images.py',
                                            convolve_func_name='numpy', write=True,parallel=True)


@protocolize()
def ext_eval_ht_l1_random_renderman(depends_on=('../config/renderman_tasks_for_ht.py',
                                                  '../config/ht_l1_random_models.py',
                                                  '../config/ten_categories_images.py')):
    """
    
    """
    protocols.extract_and_evaluate_protocol('../config/renderman_tasks_for_ht.py',
                                            '../config/ht_l1_random_models.py',
                                            '../config/ten_categories_images.py',
                                            convolve_func_name='numpy', write=True,parallel=True)
