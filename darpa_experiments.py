from starflow.protocols import Apply, Applies, protocolize

import deploy
import pythor_protocols as protocols

@Applies(deploy.images,args=('../config/darpa_heli_200_2500.py',False,('../../darpa/helidata/',)))
def generate_darpa_images():
    Apply()


@protocolize()
def make_darpa_test_models(depends_on='../config/darpa_test_models.py'):
    """
    """
    protocols.model_protocol(depends_on,parallel=False,write=True)
            

@protocolize()
def extract_darpa_test_models(depends_on=('../config/darpa_test_extraction.py',
                                          '../config/darpa_test_models.py',
                                          '../config/darpa_heli_200_2500.py')):
    a,b,c = depends_on
    protocols.extraction_protocol(a,b,c,convolve_func_name='numpy', write=True,parallel=True,save_to_db=True,batch_size=100)



@Applies(deploy.images,args=('../config/darpa_heli_200_2500_enriched.py',False,('../../darpa/helidata/',)))
def generate_darpa_enriched_images():
    Apply()


@protocolize()
def make_darpa_models(depends_on='../config/darpa_models.py'):
    protocols.model_protocol(depends_on,parallel=False,write=True)

@protocolize()
def extract_darpa_models(depends_on=('../config/darpa_extraction.py',
                                     '../config/darpa_models.py',
                                     '../config/darpa_heli_200_2500_enriched.py')):
    a,b,c = depends_on
    protocols.extraction_protocol(a,b,c,convolve_func_name='numpy', write=True,
                                  parallel=True, save_to_db=True, batch_size=100)


@protocolize()
def screen_darpa_models(depends_on=('../config/darpa_screen_evaluation.py',
                                    '../config/darpa_extraction.py', 
                                    '../config/darpa_models.py',
                                    '../config/darpa_heli_200_2500_enriched.py')):
    """
            
    """
    a,b,c,d = depends_on
    protocols.evaluate_protocol(a,b,c,d, write=True,parallel=True,use_db = True)




@Applies(deploy.images,args=('../config/darpa_heli_optimalbbox_100000_enriched.py',False,('../../darpa/helidata/',)))
def generate_darpa_optimalbbox_images():
    Apply()

@protocolize()
def make_optimal_darpa_models(depends_on='../config/darpa_optimal_models.py'):
    protocols.model_protocol(depends_on,parallel=False,write=True)

@protocolize()
def extract_optimal_darpa_models(depends_on=('../config/darpa_extraction.py',
                                            '../config/darpa_optimal_models.py',
                                            '../config/darpa_heli_optimalbbox_100000_enriched.py')):
    a,b,c = depends_on
    protocols.extraction_protocol(a,b,c,convolve_func_name='numpy', write=True,parallel=True, save_to_db=True, batch_size=100)


@protocolize()
def train_optimal_darpa_models(depends_on=('../config/darpa_optimal_training.py',
                                          '../config/darpa_extraction.py',
                                          '../config/darpa_optimal_models.py',
                                          '../config/darpa_heli_optimalbbox_100000_enriched.py')):
    a,b,c,d = depends_on
    protocols.evaluate_protocol(a,b,c,d, write=True,parallel=False,use_db = True)

@protocolize()
def train_optimal_darpa_models_more_enriched(depends_on=('../config/darpa_optimal_training_more_enriched.py',
                                          '../config/darpa_extraction.py',
                                          '../config/darpa_optimal_models.py',
                                          '../config/darpa_heli_optimalbbox_100000_enriched.py')):
    a,b,c,d = depends_on

    protocols.evaluate_protocol(a,b,c,d, write=True,parallel=False,use_db = True)


@Applies(deploy.images,args=('../config/darpa_heli_test_optimalbbox_gridded.py',False,('../../darpa/helidata_test/',)))
def generate_darpa_test_optimalbbox_images():
    Apply()


@Applies(deploy.images,args=('../config/darpa_heli_optimalbbox_gridded.py',False,('../../darpa/helidata/',)))
def generate_darpa_train_gridded_optimalbbox_images():
    Apply()
    

@protocolize()
def extract_optimal_darpa_models_ontest(depends_on=('../config/darpa_extraction.py',
                                                    '../config/darpa_optimal_models.py',
                                                    '../config/darpa_heli_test_optimalbbox_gridded.py')):
    a,b,c = depends_on
    protocols.extraction_protocol(a,b,c,convolve_func_name='numpy', write=True,parallel=True, save_to_db=True, batch_size=100)
            

@Applies(deploy.images,args=('../config/darpa_heli_test_all_optimalbbox_gridded.py',False,('../../darpa/helidata_test_all/',)))
def generate_darpa_test_all_optimalbbox_images():
    Apply()


@protocolize()
def extract_optimal_darpa_models_ontest_all(depends_on=('../config/darpa_extraction.py',
                                                    '../config/darpa_optimal_models.py',
                                                    '../config/darpa_heli_test_all_optimalbbox_gridded.py')):
    a,b,c = depends_on
    protocols.extraction_protocol(a,b,c,convolve_func_name='numpy', write=True,parallel=True, save_to_db=True, batch_size=200)


@Applies(deploy.images,args=('../config/darpa_heli_test_all_irobot.py',False,('../../darpa/helidata_test_all/','../../darpa/nv2_detections/')))
def generate_darpa_test_all_optimalbbox_images_irobot():
    Apply()


@protocolize()
def extract_optimal_darpa_models_ontest_irobot(depends_on=('../config/darpa_extraction.py',
                                                        '../config/darpa_optimal_models.py',
                                                        '../config/darpa_heli_test_all_irobot.py')):
    a,b,c = depends_on
    protocols.extraction_protocol(a,b,c,convolve_func_name='numpy', write=True,parallel=True, save_to_db=True, batch_size=200)

@protocolize()
def evaluate_optimal_darpa_models_ontest_irobot(depends_on=('../config/darpa_irobot_tasks.py',
                                                             '../config/darpa_extraction.py',
                                                             '../config/darpa_optimal_models.py',
                                                             '../config/darpa_heli_test_all_irobot.py')):
    a,b,c,d = depends_on

    protocols.evaluate_protocol(a,b,c,d, write=True,parallel = False, use_db = True)
